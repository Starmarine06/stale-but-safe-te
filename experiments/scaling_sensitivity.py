"""Does the conclusion depend on how real traces are scaled to a target load? (median vs mean vs p95 of ECMP max-util)"""
import numpy as np
import yaml

from stale_but_safe.baselines.heuristics import TunedHeuristic
from stale_but_safe.env import real
from stale_but_safe.env.te_env import EnvCfg
from stale_but_safe.env.telemetry import PROFILES
from stale_but_safe.eval.compare import episode_stats
from stale_but_safe.models.learned import LearnedPolicy, load

import sys
tuned = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "configs/tuned_heuristic.yaml"))
models = [load(p) for p in ["runs/ours_dfm_final.pt", "runs/ours_dfm_final_s1.pt", "runs/ours_dfm_final_s2.pt", "runs/ours_dfm_final_s3.pt"]]
print("| network | scaling reference | telemetry | tuned gain % | learned gain % (mean of 4 seeds) | tuned p99 regret | learned p99 regret |")
print("|---|---|---|---|---|---|---|")
for net in ("abilene", "geant", "germany17"):
    topo, trace = real.load(net)
    for stat in ("median", "mean", "p95"):
        for prof in ("ideal", "degraded"):
            cfg = EnvCfg(load=0.8, horizon=100, telemetry=PROFILES[prof], trace=trace, trace_stat=stat)
            res = {}
            for name, pols in (("tuned", [TunedHeuristic(**tuned[prof]["params"])]), ("learned", [LearnedPolicy(m, c, "g") for m, c in models])):
                g, rg = [], []
                for pol in pols:
                    rows = [episode_stats(topo, cfg, pol, 11000 + s) for s in range(6)]
                    g.append(np.mean([1 - r["mu"] / r["ecmp_mu"] for r in rows])); rg.append(np.quantile(np.concatenate([r["regret"] for r in rows]), .99))
                res[name] = (100 * np.mean(g), np.mean(rg))
            print(f"| {net} | {stat} | {prof} | {res['tuned'][0]:.1f} | {res['learned'][0]:.1f} | {res['tuned'][1]:+.3f} | {res['learned'][1]:+.3f} |", flush=True)
