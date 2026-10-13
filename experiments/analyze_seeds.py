"""Seed-to-seed variance of the headline comparison (does the ranking flip survive retraining?).

  python experiments/analyze_seeds.py            # reads experiments/final/results_{ours,ideal,seed1..}.json
Seed 0 = results_ours.json (gnn-ours) + results_ideal.json (gnn-ideal); seeds s>=1 = results_seed{s}.json.
Per training seed and profile: mean dScore over test topologies, mean p99 regret, win-rate vs tuned.
"""
import glob
import json
import re
from collections import defaultdict

import numpy as np

F = "experiments/final/"
rows = defaultdict(dict)      # (profile, model, seed) -> {topo: rec}
tuned = defaultdict(dict)
for p in [F + "results_ours.json", F + "results_ideal.json"] + sorted(glob.glob(F + "results_seed*.json")):
    seed = int(m.group(1)) if (m := re.search(r"seed(\d+)", p)) else 0
    for r in json.load(open(p))["results"]:
        if r["method"] in ("gnn-ours", "gnn-ideal"):
            rows[(r["profile"], r["method"], seed)][r["topo"]] = r
        elif r["method"] == "tuned":
            tuned[r["profile"]][r["topo"]] = r

print("| profile | model | seeds | dScore per seed | mean ± SD | p99 regret mean | wins vs tuned (per seed) |")
print("|---|---|---|---|---|---|---|")
for prof in ("ideal", "rich", "snmp", "degraded"):
    for model in ("gnn-ours", "gnn-ideal"):
        seeds = sorted(s for (p, m, s) in rows if p == prof and m == model)
        if not seeds:
            continue
        ds, p99, wins = [], [], []
        for s in seeds:
            rec = rows[(prof, model, s)]
            common = sorted(set(rec) & set(tuned[prof]))
            d = np.array([rec[t]["delta"] for t in common])
            ds.append(d.mean()); p99.append(np.mean([rec[t]["regret_p99"] for t in common]))
            wins.append(int(sum(rec[t]["delta"] < tuned[prof][t]["delta"] for t in common)))
        print(f"| {prof} | {model} | {len(seeds)} | {', '.join(f'{x:+.3f}' for x in ds)} | {np.mean(ds):+.3f} ± {np.std(ds, ddof=1) if len(ds) > 1 else float('nan'):.3f} "
              f"| {np.mean(p99):.3f} | {'/'.join(map(str, wins))} of {len(common)} |")
