"""Final evaluation per docs/eval_protocol.md on the held-out Zoo test_subset (+ Abilene).

  python experiments/final_eval.py --ours runs/ours_dfm_final.pt --ideal runs/ideal_dfm_final.pt \
        --out experiments/final/results.json [--seeds 5] [--topos N] [--horizon 100]

Writes per-(profile, method, topology) paired means so statistics can be computed over topologies.
The LP oracle is run only on graphs with n <= --oracle-max-n (it is O(D*k) per step).
"""
import argparse
import json
import time

import numpy as np
import yaml

from stale_but_safe.baselines.heuristics import ECMP, LPOracle, TunedHeuristic
from stale_but_safe.env.te_env import EnvCfg
from stale_but_safe.env.telemetry import PROFILES
from stale_but_safe.env.topology import abilene
from stale_but_safe.env.zoo import load_split, zoo_topology
from stale_but_safe.eval.compare import episode_stats, ecmp_score, score
from stale_but_safe.guard.guard import Guard, GuardCfg

ap = argparse.ArgumentParser()
ap.add_argument("--ours")
ap.add_argument("--ideal")
ap.add_argument("--out", required=True)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--topos", type=int, default=0, help="limit number of test graphs (0 = all)")
ap.add_argument("--horizon", type=int, default=100)
ap.add_argument("--load", type=float, default=0.8)
ap.add_argument("--eps", type=float, default=0.0)
ap.add_argument("--profiles", default="ideal,rich,snmp,degraded")
ap.add_argument("--oracle-max-n", type=int, default=25)
ap.add_argument("--only-models", action="store_true", help="skip ecmp/tuned/oracle (already run)")
ap.add_argument("--guard-layers", default="123", help="subset of 1,2,3, e.g. 2 = pre-action check only")
ap.add_argument("--include-tuned-guard", action="store_true")
ap.add_argument("--tuned-config", default="configs/tuned_heuristic.yaml")
ap.add_argument("--tuned-only", action="store_true", help="run only tuned and tuned+guard")
ap.add_argument("--scenario", default="gravity", choices=["gravity", "hotspot", "shift", "flash", "fail"])
a = ap.parse_args()

tuned = yaml.safe_load(open(a.tuned_config))
GC = lambda: GuardCfg(eps=a.eps, layers=tuple(str(i) in a.guard_layers for i in (1, 2, 3)))
names = load_split("test_subset")
topos = [zoo_topology(n) for n in names]
if a.topos:
    topos = topos[:a.topos]

models = {}
if a.ours or a.ideal:
    from stale_but_safe.models.learned import LearnedPolicy, load
    for key, path in (("ours", a.ours), ("ideal", a.ideal)):
        if path:
            models[key] = load(path)


def traffic_cfg(h):
    return {"gravity": {}, "hotspot": {"hotspot_prob": 0.1}, "shift": {"shift_at": h // 2},
            "flash": {"flash_at": h // 3, "flash_len": 20}, "fail": {}}[a.scenario]


def learned(key):
    from stale_but_safe.models.learned import LearnedPolicy
    m, c = models[key]
    return LearnedPolicy(m, c, f"gnn-{key}")


methods = {"ecmp": lambda p: ECMP(), "tuned": lambda p: TunedHeuristic(**tuned[p]["params"]),
           "tuned+guard": lambda p: Guard(TunedHeuristic(**tuned[p]["params"]), GC())}
for key in models:
    methods[f"gnn-{key}"] = (lambda k: lambda p: learned(k))(key)
    methods[f"gnn-{key}+guard"] = (lambda k: lambda p: Guard(learned(k), GC()))(key)

out = {"args": vars(a), "results": []}
t0 = time.time()
seeds = [5000 + i for i in range(a.seeds)]
for t in topos:
    for pn in a.profiles.split(","):
        cfg = EnvCfg(load=a.load, horizon=a.horizon, telemetry=PROFILES[pn], traffic=traffic_cfg(a.horizon),
                     fail_at=int(a.horizon * 0.4) if a.scenario == "fail" else None)
        if a.tuned_only:
            todo = {k: v for k, v in methods.items() if k.startswith("tuned")}
        else:
          todo = {k: v for k, v in methods.items() if not a.only_models or k.startswith("gnn") or (a.include_tuned_guard and k == "tuned+guard")}
        if t.n <= a.oracle_max_n and not a.only_models and not a.tuned_only:
            todo["oracle"] = lambda p: LPOracle()
        for mname, mk in todo.items():
            pol = mk(pn)
            rows = [episode_stats(t, cfg, pol, s) for s in seeds]
            reg = np.concatenate([r["regret"] for r in rows])
            rec = {"topo": t.name, "n": t.n, "profile": pn, "method": mname,
                   "delta": float(np.mean([score(r) - ecmp_score(r) for r in rows])),
                   "gain": float(np.mean([1 - r["mu"] / r["ecmp_mu"] for r in rows])),
                   "loss": float(np.mean([r["loss"] for r in rows])),
                   "churn": float(np.mean([r["churn"] for r in rows])),
                   "regret_p99": float(np.quantile(reg, 0.99)), "regret_mean": float(reg.mean()),
                   "regret_cvar95": float(np.sort(reg)[-max(1, len(reg) // 20):].mean())}
            if hasattr(pol, "stats"):
                rec["fallback_rate"] = pol.stats["fallback"] / max(pol.stats["steps"], 1)
            out["results"].append(rec)
        print(f"{t.name:22s} n={t.n:2d} {pn:8s} done t={time.time()-t0:.0f}s", flush=True)
        json.dump(out, open(a.out, "w"))
json.dump(out, open(a.out, "w"))
