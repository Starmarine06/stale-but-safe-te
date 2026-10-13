"""DOTE-style per-network baseline vs zero-shot GNN vs tuned (temporal test split)."""
import glob, json, sys
from collections import defaultdict
import numpy as np
by = defaultdict(dict)
for p in sorted(glob.glob("experiments/final/dote_*.json")) + sorted(glob.glob("experiments/final/static_*.json")):
    for r in json.load(open(p))["results"]:
        by[(r["topo"], r["profile"])][r["method"]] = r
print("| network (test split) @load | telemetry | tuned | DOTE-style (ideal-trained) | DOTE-style (degraded-trained) | ours GNN, mean+-SD of 4 seeds | static LP (mean TM) | winner |")
print("|---|---|---|---|---|---|---|---|")
for (topo, prof), m in sorted(by.items(), key=lambda kv: (kv[0][0], ["ideal", "snmp", "degraded"].index(kv[0][1]))):
    if "dote-degraded" not in m: continue
    g = np.array([100 * m[k]["gain"] for k in m if k.startswith("gnn-ours")])
    vals = {"tuned": 100 * m["tuned"]["gain"], "dote-ideal": 100 * m["dote-ideal"]["gain"], "dote-degraded": 100 * m["dote-degraded"]["gain"], "ours": g.mean(), "static": 100 * m["static-lp"]["gain"] if "static-lp" in m else -1e9}
    best = max(vals, key=vals.get)
    print(f"| {topo} | {prof} | {vals['tuned']:.1f} / {m['tuned']['regret_p99']:+.3f} | {vals['dote-ideal']:.1f} / {m['dote-ideal']['regret_p99']:+.3f} "
          f"| {vals['dote-degraded']:.1f} / {m['dote-degraded']['regret_p99']:+.3f} | {g.mean():.1f}+-{g.std(ddof=1):.1f} | {vals['static']:.1f} | {best} |")
print("\n(cells: gain % vs ECMP / p99 regret)")
