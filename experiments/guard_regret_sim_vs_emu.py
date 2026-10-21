"""Guard (eps 0.05, all layers) on the nine TCP scenarios: simulator replay vs TCP emulation, like for like.
Every number compares full trajectories: guarded run vs unguarded learned run vs ECMP run, capped max-utilisation
(the emulator's counters saturate at 1), degraded telemetry, intervals 3 onward.  Regret = policy - ECMP in the same interval.
(An earlier check compared the guarded run with the raw proposal evaluated on the guard's own trajectory; that understates the
guard's cost and should not be used.)
  python experiments/guard_regret_sim_vs_emu.py"""
import json
import sys
from pathlib import Path
import numpy as np

sys.argv = ["x"]
exec(open("experiments/guard_carried_replay.py", encoding="utf-8").read().split('print("| guard | counters')[0])
from stale_but_safe.baselines.heuristics import ECMP
from stale_but_safe.emu.run_emulation import tag_of


def traj(n, pol_fn):
    d = Path("runs/emu_tcp") / n; sc = S.load(d / "scenario.json")
    mp = (d / "model.txt").read_text().strip() if (d / "model.txt").exists() else f"runs/ours_dfm_final_{n}.pt"
    m, c = load(mp); topo = S.topo_from_dict(sc["topo"])
    env = TEEnv(topo, S.sim_cfg(sc, PROFILES["degraded"], carried=True), sc["seed"]); obs = env.reset(sc["seed"])
    pol = pol_fn(m, c); pol.reset(env); mus = []; done = False
    while not done:
        dem, up = env.peek_demand(), env.link_up.copy(); act = pol.act(obs, env); mus.append(min(mu(topo, dem, act, up), 1.0))
        obs, _, done, _ = env.step(act)
    return np.array(mus)[3:]


def emu(n, spec, prof):
    d = Path("runs/emu_tcp") / n; t = tag_of(spec, prof)
    r = json.loads((d / t / f"ctrl_{t}.json").read_text())["rows"]
    return {q["interval"]: q["true_max_util"] for q in r if q["interval"] >= 3 and q["latency_ms"] > 0}


sim = {k: [] for k in ("learned", "guard")}; em = {k: [] for k in ("learned", "guard")}
for n in NAMES:
    mp = Path("runs/emu_tcp") / n / "model.txt"; M = mp.read_text().strip() if mp.exists() else f"runs/ours_dfm_final_{n}.pt"
    E = traj(n, lambda m, c: ECMP())
    sim["learned"].append(traj(n, lambda m, c: LearnedPolicy(m, c, "gnn")) - E)
    sim["guard"].append(traj(n, lambda m, c: Guard(LearnedPolicy(m, c, "gnn"), GuardCfg(eps=0.05))) - E)
    e = emu(n, "ecmp", "ideal")
    for k, sp in (("learned", f"model:{M}"), ("guard", f"guard0.05@model:{M}")):
        x = emu(n, sp, "degraded"); ks = sorted(set(x) & set(e)); em[k].append(np.array([min(x[i], 1) - min(e[i], 1) for i in ks]))
print("| setting | policy | mean regret vs ECMP | p90 | p99 | intervals worse than ECMP by >0.02 |\n|---|---|---|---|---|---|")
for lab, D in (("simulator replay", sim), ("TCP emulation", em)):
    for k in ("learned", "guard"):
        a = np.concatenate(D[k])
        print(f"| {lab} | {k} | {a.mean():+.4f} | {np.quantile(a, .9):+.4f} | {np.quantile(a, .99):+.4f} | {100 * np.mean(a > .02):.1f}% |")
for lab, D in (("simulator replay", sim), ("TCP emulation", em)):
    d = np.array([g.mean() - l.mean() for g, l in zip(D["guard"], D["learned"])])
    print(f"{lab}: guard - learned, mean over scenarios {d.mean():+.4f}, guard higher in {(d > 0).sum()}/9")
