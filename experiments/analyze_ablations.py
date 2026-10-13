"""Ablations (one training seed each) vs the full model, on the 24 held-out Zoo graphs, ideal and degraded telemetry."""
import json
from collections import defaultdict
import numpy as np
F = "experiments/final/"
full = defaultdict(dict)
for r in json.load(open(F + "results_ours.json"))["results"]:
    if r["method"] in ("gnn-ours", "tuned"):
        full[(r["method"], r["profile"])][r["topo"]] = r
names = {"noh": "no history (H=1)", "noage": "no telemetry-age input", "nognn": "no GNN (per-link MLP)", "nomodel": "no fluid-model features"}
print("| model | telemetry | dScore (mean over 24 graphs) | p99 regret | graphs better than tuned | delta vs full model |")
print("|---|---|---|---|---|---|")
for prof in ("ideal", "degraded"):
    ful = full[("gnn-ours", prof)]; tun = full[("tuned", prof)]
    topos = sorted(ful)
    d0 = np.array([ful[t]["delta"] for t in topos])
    print(f"| full model (seed 0) | {prof} | {d0.mean():+.3f} | {np.mean([ful[t]['regret_p99'] for t in topos]):.3f} | {sum(ful[t]['delta'] < tun[t]['delta'] for t in topos)} of {len(topos)} | - |")
    for k, nm in names.items():
        rows = {r["topo"]: r for r in json.load(open(F + f"results_abl_{k}.json"))["results"] if r["method"] == "gnn-ours" and r["profile"] == prof}
        common = [t for t in topos if t in rows]
        d = np.array([rows[t]["delta"] for t in common])
        diff = d - np.array([ful[t]["delta"] for t in common])
        rng = np.random.default_rng(0)
        bs = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
        print(f"| {nm} | {prof} | {d.mean():+.3f} | {np.mean([rows[t]['regret_p99'] for t in common]):.3f} | "
              f"{sum(rows[t]['delta'] < tun[t]['delta'] for t in common)} of {len(common)} | {diff.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] |")
