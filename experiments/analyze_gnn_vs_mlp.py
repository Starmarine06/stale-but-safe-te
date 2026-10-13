"""Rigorous GNN vs per-link-MLP (and fluid-features vs none) comparison on the 24 held-out Zoo graphs.
Unit = graph; each model's score on a graph is the mean over evaluation seeds; compare training-seed means per graph.
Full model: seeds 0-3 (results_ours.json for s0, results_seed{1,2,3}.json 'gnn-ours'); ablations: results_abl_{x}.json (s0), results_abl_{x}_s{n}.json."""
import glob, json, re
from collections import defaultdict
import numpy as np
F = "experiments/final/"
def load(path, method="gnn-ours"):
    d = defaultdict(dict)
    for r in json.load(open(path))["results"]:
        if r["method"] == method:
            d[r["profile"]][r["topo"]] = r
    return d
models = {"full": [load(F + "results_ours.json")] + [load(F + f"results_seed{s}.json") for s in (1, 2, 3) if glob.glob(F + f"results_seed{s}.json")]}
for x in ("nognn", "nomodel"):
    models[x] = [load(F + f"results_abl_{x}.json")] + [load(p) for p in sorted(glob.glob(F + f"results_abl_{x}_s*.json"))]
rng = np.random.default_rng(0)
print("| telemetry | model | training seeds | dScore per seed | mean | p99 regret mean |")
print("|---|---|---|---|---|---|")
per = {}
for prof in ("ideal", "degraded"):
    for k, runs in models.items():
        ok = [r for r in runs if len(r[prof]) >= 24]
        if not ok: continue
        topos = sorted(ok[0][prof])
        arr = np.array([[r[prof][t]["delta"] for t in topos] for r in ok])          # seeds x graphs
        per[(prof, k)] = arr
        p99 = np.mean([[r[prof][t]["regret_p99"] for t in topos] for r in ok])
        print(f"| {prof} | {k} | {len(ok)} | {', '.join(f'{a.mean():+.3f}' for a in arr)} | {arr.mean():+.3f} | {p99:.3f} |")
print("\nPaired over graphs (seed-averaged per graph): difference ablation - full model, bootstrap 95% CI over graphs; negative = ablation better")
for prof in ("ideal", "degraded"):
    for k in ("nognn", "nomodel"):
        if (prof, k) in per and (prof, "full") in per:
            d = per[(prof, k)].mean(0) - per[(prof, "full")].mean(0)
            bs = [rng.choice(d, len(d)).mean() for _ in range(4000)]
            print(f"  {prof:9s} {k:8s} (n_seeds={per[(prof, k)].shape[0]} vs {per[(prof, 'full')].shape[0]}): {d.mean():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]; ablation better on {int((d < 0).sum())}/{len(d)} graphs")
