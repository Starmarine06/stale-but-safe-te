"""Compare learned (4 training seeds, mean per graph) with the original and re-tuned per-profile heuristics on the 24 Zoo test graphs.
  python experiments/analyze_retune.py > experiments/final/table_retune.md"""
import glob
import json
import re
from collections import defaultdict

import numpy as np

F = "experiments/final/"
raw = json.load(open(F + "retune_profiles.json"))
L = defaultdict(lambda: defaultdict(list)); old = defaultdict(dict)
for p in [F + "results_ours.json"] + sorted(glob.glob(F + "results_seed*.json")):
    for r in json.load(open(p))["results"]:
        if r["method"] == "gnn-ours":
            L[r["profile"]][r["topo"]].append(r)
for r in json.load(open(F + "results_ours.json"))["results"]:
    if r["method"] == "tuned":
        old[r["profile"]][r["topo"]] = r
rng = np.random.default_rng(0)
def ci(x):
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(4000)]
    return np.percentile(b, [2.5, 97.5])
print("dScore vs ECMP (lower = better); 24 held-out Zoo graphs; learned = mean of 4 training seeds. 'diff' = heuristic - learned, so positive = learned better;\n"
      "95% bootstrap CI over graphs. W/T/L = graphs where the learned policy wins/ties(|diff|<0.002)/loses. Re-tuned = 5 independent tuner seeds x 80 trials.\n")
print("| profile | heuristic | dScore | p99 regret | diff vs learned [95% CI] | learned W/T/L |\n|---|---|---|---|---|---|")
for prof in ("ideal", "rich", "snmp", "degraded"):
    topos = sorted(L[prof]); ld = np.array([np.mean([r["delta"] for r in L[prof][t]]) for t in topos])
    lp = np.mean([np.mean([r["regret_p99"] for r in L[prof][t]]) for t in topos])
    print(f"| {prof} | learned (4 seeds) | {ld.mean():+.3f} | {lp:.3f} | - | - |")
    def row(name, d, p99):
        diff = d - ld; lo, hi = ci(diff)
        w = int((diff > 0.002).sum()); l = int((diff < -0.002).sum())
        print(f"| {prof} | {name} | {d.mean():+.3f} | {p99:.3f} | {diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | {w}/{len(diff)-w-l}/{l} |")
    row("original (40 trials, 1 run)", np.array([old[prof][t]["delta"] for t in topos]), np.mean([old[prof][t]["regret_p99"] for t in topos]))
    rs = [r for r in raw if r["profile"] == prof]
    for r in sorted(rs, key=lambda r: r["k"]):
        pass
    ds = [np.array([r["per_topo"][t]["delta"] for t in topos]) for r in rs]
    p9 = [np.mean([r["per_topo"][t]["regret_p99"] for t in topos]) for r in rs]
    row("re-tuned, mean over 5 tuner seeds", np.mean(ds, 0), np.mean(p9))
    sel = min(range(len(rs)), key=lambda i: rs[i]["tune_score"])
    row("re-tuned, selected by tuning score", ds[sel], p9[sel])
    print(f"| {prof} | re-tuned, per tuner seed (dScore) | {', '.join(f'{d.mean():+.3f}' for d in ds)} | | | |")
