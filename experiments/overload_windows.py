"""Typical vs overloaded trace windows (real Abilene): where does each method help or hurt?
Overloaded window := ECMP's mean max-util over the window > 2 (traffic far above the requested load, e.g. surges).
  python experiments/overload_windows.py  [network] [load]"""
import json
import sys

import numpy as np
import yaml

from stale_but_safe.baselines.heuristics import ECMP, TunedHeuristic
from stale_but_safe.env import real
from stale_but_safe.env.te_env import EnvCfg
from stale_but_safe.env.telemetry import PROFILES
from stale_but_safe.eval.runner import rollout
from stale_but_safe.guard.guard import Guard, GuardCfg
from stale_but_safe.models.learned import LearnedPolicy, load

net = sys.argv[1] if len(sys.argv) > 1 else "abilene"
ld = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
topo, trace = real.load(net)
tuned = yaml.safe_load(open("configs/tuned_heuristic.yaml"))
models = {f"learned-s{i}": load(p) for i, p in enumerate(["runs/ours_dfm_final.pt", "runs/ours_dfm_final_s1.pt", "runs/ours_dfm_final_s2.pt", "runs/ours_dfm_final_s3.pt"])}
G2 = GuardCfg(eps=-0.06, layers=(True, True, False))
seeds = list(range(8000, 8030))
rows = {}
for prof in ("ideal", "degraded"):
    cfg = EnvCfg(load=ld, horizon=100, telemetry=PROFILES[prof], trace=trace)
    for s in seeds:
        e = rollout(topo, cfg, ECMP(), s)["ecmp_max_util"].mean()
        rec = {"ecmp_mu": float(e)}
        pols = {"tuned": TunedHeuristic(**tuned[prof]["params"]),
                "learned-s0+guard": Guard(LearnedPolicy(*models["learned-s0"], "g"), G2)}
        pols.update({k: LearnedPolicy(*v, k) for k, v in models.items()})
        for k, p in pols.items():
            rec[k] = float(rollout(topo, cfg, p, s)["max_util"].mean())
        rows[(prof, s)] = rec
    print(prof, "done", flush=True)
out = {f"{p}-{s}": r for (p, s), r in rows.items()}
json.dump(out, open(f"experiments/final/overload_{net}_{ld}.json", "w"))
print("| telemetry | window type | windows | tuned gain % (median, worst) | learned gain % (median over seeds, worst) | learned s0+guard (median, worst) | learned worse than ECMP in |")
print("|---|---|---|---|---|---|---|")
for prof in ("ideal", "degraded"):
    for label, sel in (("typical (ECMP mean max-util <= 2)", lambda e: e <= 2), ("overloaded (> 2)", lambda e: e > 2)):
        keys = [(prof, s) for s in seeds if sel(rows[(prof, s)]["ecmp_mu"])]
        if not keys:
            print(f"| {prof} | {label} | 0 | - | - | - | - |"); continue
        g = lambda k: np.array([100 * (1 - rows[key][k] / rows[key]["ecmp_mu"]) for key in keys])
        L = np.mean([g(f"learned-s{i}") for i in range(4)], 0)
        worse = np.mean([ (g(f"learned-s{i}") < 0).mean() for i in range(4)])
        print(f"| {prof} | {label} | {len(keys)} | {np.median(g('tuned')):.1f}, {g('tuned').min():.1f} | {np.median(L):.1f}, {L.min():.1f} "
              f"| {np.median(g('learned-s0+guard')):.1f}, {g('learned-s0+guard').min():.1f} | {100 * worse:.0f}% of windows |")
