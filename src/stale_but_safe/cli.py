"""stale-but-safe command line.

  stale-but-safe demo --topology abilene --mode shadow|active [--model runs/ours.pt] [--profile snmp]
  stale-but-safe scenario --out runs/emu/compose [--topology emu10] [--steps 60]
  stale-but-safe audit --network abilene --polling 5 --delay 1 --loss 0.02 --out reports/abilene
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from .baselines.heuristics import ECMP, TunedHeuristic
from .env.te_env import EnvCfg, TEEnv
from .env.telemetry import PROFILES
from .env.topology import abilene, synthetic
from .eval.runner import rollout
from .guard.guard import Guard, GuardCfg


def get_topology(name: str):
    if name == "abilene":
        return abilene()
    if name == "emu10":
        return synthetic("ba", 10, seed=7777)
    if name.startswith(("ba", "er", "geo")):
        kind = name.rstrip("0123456789")
        return synthetic(kind, int(name[len(kind):]), seed=1)
    from .env.zoo import zoo_topology
    return zoo_topology(name)


def make_policy(a, topo):
    prof = a.profile
    if a.model:
        from .models.learned import LearnedPolicy, load
        m, c = load(a.model)
        pol = LearnedPolicy(m, c, "gnn")
    else:
        tp = yaml.safe_load(open(Path(__file__).resolve().parents[2] / "configs" / "tuned_heuristic_v2.yaml"))
        pol = TunedHeuristic(**tp[prof]["params"])
    return Guard(pol, GuardCfg(eps=a.eps)) if not a.no_guard else pol


class Shadow:
    """Shadow mode: the policy proposes, ECMP is what actually runs. Reports what the
    policy *would* have predicted (fluid model on the estimated state) vs ECMP."""

    def __init__(self, pol):
        self.pol, self.name = pol, pol.name + "[shadow]"
        self.pred = []

    def reset(self, env):
        self.pol.reset(env)
        self.ecmp = ECMP(); self.ecmp.reset(env)

    def act(self, obs, env):
        prop = self.pol.act(obs, env)
        base = self.ecmp.act(obs, env)
        topo, dem, up = env.topo, obs["demand"], obs["link_up"]
        f = lambda s: float((topo.inc @ (dem[:, None] * s).ravel() / topo.cap)[up].max())
        self.pred.append((f(prop), f(base)))
        return base


def cmd_demo(a):
    topo = get_topology(a.topology)
    cfg = EnvCfg(load=a.load, horizon=a.steps, telemetry=PROFILES[a.profile])
    pol = make_policy(a, topo)
    base = rollout(topo, cfg, ECMP(), a.seed)
    if a.mode == "shadow":
        sh = Shadow(pol)
        rollout(topo, cfg, sh, a.seed)
        p = np.array(sh.pred)
        print(f"[shadow] topology={topo.name} profile={a.profile}: policy would predict "
              f"max-util {p[:, 0].mean():.3f} vs ECMP {p[:, 1].mean():.3f} "
              f"({100 * (1 - p[:, 0].mean() / p[:, 1].mean()):.1f}% lower); network ran ECMP.")
        return
    r = rollout(topo, cfg, pol, a.seed)
    print(f"[active] topology={topo.name} profile={a.profile} policy={pol.name}")
    print(f"  mean max-util: {r['max_util'].mean():.3f}   ECMP: {base['max_util'].mean():.3f}   "
          f"gain: {100 * (1 - r['max_util'].mean() / base['max_util'].mean()):.1f}%")
    print(f"  loss: {r['loss'].mean():.4f} (ECMP {base['loss'].mean():.4f})   "
          f"p99 regret vs ECMP: {np.quantile(r['regret'], .99):.3f}")
    if hasattr(pol, "stats"):
        s = pol.stats
        print(f"  guard: {s['fallback']}/{s['steps']} intervals on ECMP "
              f"(L1={s['L1']} L2={s['L2']} L3={s['L3']})")


def cmd_scenario(a):
    from .emu import scenario as S
    topo = get_topology(a.topology)
    cfg = EnvCfg(load=a.load, horizon=a.steps)
    sc = S.make_scenario(topo, cfg, a.seed, a.steps, active_n=a.active, delta=a.delta, mbps_per_unit=a.scale)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    S.save(sc, out / "scenario.json")
    print(f"wrote {out / 'scenario.json'} ({topo.name}, {len(sc['active'])} flows, {a.steps} intervals)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="stale-but-safe")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo")
    d.add_argument("--topology", default="abilene")
    d.add_argument("--mode", choices=["shadow", "active"], default="active")
    d.add_argument("--model")
    d.add_argument("--profile", default="snmp", choices=list(PROFILES))
    d.add_argument("--eps", type=float, default=0.05)
    d.add_argument("--no-guard", action="store_true")
    d.add_argument("--load", type=float, default=0.8)
    d.add_argument("--steps", type=int, default=100)
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(fn=cmd_demo)
    s = sub.add_parser("scenario")
    s.add_argument("--out", required=True)
    s.add_argument("--topology", default="emu10")
    s.add_argument("--steps", type=int, default=60)
    s.add_argument("--active", type=int, default=30)
    s.add_argument("--delta", type=float, default=1.0)
    s.add_argument("--scale", type=float, default=4.0)
    s.add_argument("--load", type=float, default=0.8)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_scenario)
    u = sub.add_parser("audit", help="is TE worth it on this network, and does it survive your telemetry?")
    u.add_argument("--network", required=True)
    u.add_argument("--traffic", help="trace .npz (T x D) or long CSV t,src,dst,value")
    u.add_argument("--profile", choices=list(PROFILES), help="shortcut instead of --polling/--delay/--noise/--loss")
    u.add_argument("--polling", type=int, default=5, help="telemetry polling period, in TE intervals")
    u.add_argument("--delay", type=int, default=1, help="telemetry delay, in intervals")
    u.add_argument("--noise", type=float, default=0.05, help="relative measurement noise")
    u.add_argument("--loss", type=float, default=0.02, help="probability a counter is missing")
    u.add_argument("--load", type=float, default=0.8, help="scale traffic so median ECMP max-util = load")
    u.add_argument("--horizon", type=int, default=100)
    u.add_argument("--seeds", type=int, default=6, help="trace windows per cell (>=5 needed for an interval)")
    u.add_argument("--model", default=str(Path(__file__).resolve().parents[2] / "runs" / "ours_dfm_final.pt"))
    u.add_argument("--out", default="reports/audit")
    u.set_defaults(fn=lambda a: __import__("stale_but_safe.audit", fromlist=["main"]).main(a))
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
