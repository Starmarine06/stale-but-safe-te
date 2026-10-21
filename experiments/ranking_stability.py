"""How often does the learned-vs-heuristic verdict change with the tuner seed and the training seed?
Zoo, 24 held-out graphs, dScore (lower = better). For every profile, each of the 5 tuner seeds (80 trials each) is compared with
(a) the mean of the 4 training seeds and (b) each single training seed (20 pairs). Verdict per comparison: paired bootstrap 95% interval
over graphs (4000 resamples) of heuristic - learned dScore; L = interval above 0 (learned ahead), H = below 0 (heuristic ahead), = otherwise.
  python experiments/ranking_stability.py > experiments/final/table_ranking_stability.md"""
import glob
import json
from collections import defaultdict
import numpy as np

F = "experiments/final/"
raw = json.load(open(F + "retune_profiles.json"))
Ls = defaultdict(lambda: defaultdict(dict))        # profile -> training seed -> topo -> delta
for si, p in enumerate([F + "results_ours.json"] + sorted(glob.glob(F + "results_seed*.json"))):
    for r in json.load(open(p))["results"]:
        if r["method"] == "gnn-ours":
            Ls[r["profile"]][si][r["topo"]] = np.mean([r["delta"]]) if not isinstance(r["delta"], list) else np.mean(r["delta"])
rng = np.random.default_rng(0)


def verdict(diff):
    b = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(4000)])
    lo, hi = np.percentile(b, [2.5, 97.5])
    return "L" if lo > 0 else "H" if hi < 0 else "="


print("| profile | tuner seeds (of 5): learned ahead / tie / heuristic ahead, vs mean of training seeds | pairs (of 20): L / = / H, single training seed | mean diff range over tuner seeds |\n|---|---|---|---|")
for prof in ("ideal", "rich", "snmp", "degraded"):
    topos = sorted(next(iter(Ls[prof].values())))
    L = {s: np.array([Ls[prof][s][t] for t in topos]) for s in Ls[prof]}
    Lm = np.mean([L[s] for s in L], axis=0)
    H = [np.array([r["per_topo"][t]["delta"] for t in topos]) for r in sorted([r for r in raw if r["profile"] == prof], key=lambda r: r["k"])]
    v_mean = [verdict(h - Lm) for h in H]
    v_pair = [verdict(h - L[s]) for h in H for s in L]
    cnt = lambda v: tuple(v.count(x) for x in ("L", "=", "H"))
    a, b = cnt(v_mean), cnt(v_pair)
    d = [float((h - Lm).mean()) for h in H]
    print(f"| {prof} | {a[0]} / {a[1]} / {a[2]} | {b[0]} / {b[1]} / {b[2]} | {min(d):+.3f} to {max(d):+.3f} |")
