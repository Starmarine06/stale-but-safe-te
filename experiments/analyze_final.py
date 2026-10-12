"""Tables + statistics from final_eval JSON files (paired across topologies).

  python experiments/analyze_final.py experiments/final/results_ours.json [more.json ...] [--md out.md]
Reports per telemetry profile: mean score-vs-ECMP (dScore, lower = better) with 95% bootstrap CI over
topologies, IQM, mean gain %, mean p99 regret, mean fallback rate, and Holm-corrected paired Wilcoxon tests.
"""
import json
import sys
from collections import defaultdict

import numpy as np

from stale_but_safe.eval.stats import bootstrap_ci, iqm, wilcoxon_holm

args = [a for a in sys.argv[1:] if not a.startswith("--")]
md_out = sys.argv[sys.argv.index("--md") + 1] if "--md" in sys.argv else None
rows = []
for p in args:
    if p == md_out:
        continue
    rows += json.load(open(p))["results"]

by = defaultdict(dict)          # (profile, method) -> {topo: rec}
for r in rows:
    by[(r["profile"], r["method"])][r["topo"]] = r
profiles = [p for p in ("ideal", "rich", "snmp", "degraded") if any(k[0] == p for k in by)]
methods = sorted({k[1] for k in by}, key=lambda m: (m != "ecmp", m))
lines = []
P = lambda s="": (print(s), lines.append(s))

for pn in profiles:
    # topologies where every non-oracle method is present
    common = set.intersection(*[set(by[(pn, m)]) for m in methods if m != "oracle" and (pn, m) in by])
    common = sorted(common)
    P(f"\n### profile = {pn}  (n_topologies = {len(common)})\n")
    P("| method | dScore (95% CI) | IQM | gain % | p99 regret | loss | fallback |")
    P("|---|---|---|---|---|---|---|")
    for m in methods:
        if (pn, m) not in by:
            continue
        sel = common if m != "oracle" else sorted(set(common) & set(by[(pn, m)]))
        d = np.array([by[(pn, m)][t]["delta"] for t in sel])
        g = np.array([by[(pn, m)][t]["gain"] for t in sel])
        p99 = np.array([by[(pn, m)][t]["regret_p99"] for t in sel])
        ls = np.array([by[(pn, m)][t]["loss"] for t in sel])
        fb = [by[(pn, m)][t].get("fallback_rate") for t in sel]
        fbm = f"{np.mean([x for x in fb if x is not None]):.2f}" if any(x is not None for x in fb) else "-"
        mu, lo, hi = bootstrap_ci(d)
        tag = f"{m}" + (f" (n={len(sel)})" if m == "oracle" else "")
        P(f"| {tag} | {mu:+.3f} [{lo:+.3f}, {hi:+.3f}] | {iqm(d):+.3f} | {100*g.mean():.1f} | {p99.mean():.3f} | {ls.mean():.3f} | {fbm} |")
    # paired tests of interest
    ref = [("gnn-ours", "tuned"), ("gnn-ours", "gnn-ideal"), ("gnn-ours+guard", "gnn-ours"), ("tuned+guard", "tuned"),
           ("gnn-ours", "ecmp")]
    pairs = {}
    for a, b in ref:
        if (pn, a) in by and (pn, b) in by:
            pairs[f"{a} vs {b}"] = ([by[(pn, a)][t]["delta"] for t in common], [by[(pn, b)][t]["delta"] for t in common])
    if pairs:
        adj = wilcoxon_holm(pairs)
        P("\nPaired Wilcoxon on dScore (Holm-adjusted p): " + "; ".join(
            f"{k}: p={v:.3g} (mean diff {np.mean(np.array(pairs[k][0]) - np.array(pairs[k][1])):+.3f})" for k, v in adj.items()))

if md_out:
    open(md_out, "w").write("\n".join(lines) + "\n")
