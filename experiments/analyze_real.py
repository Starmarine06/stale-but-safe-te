"""Tables for the real-trace evaluation (experiments/final/real_<net>.json).

Per network x load x profile: gain % vs ECMP (mean-max-util reduction), p99 regret, for ecmp/tuned/learned
(mean +- SD over the training seeds)/ideal-trained/guarded/oracle.   python experiments/analyze_real.py [--md out.md]
"""
import glob
import json
import sys
from collections import defaultdict

import numpy as np

rows = []
for p in sorted(glob.glob("experiments/final/real_*.json")):
    rows += json.load(open(p))["results"]
by = defaultdict(dict)
for r in rows:
    by[(r["topo"], r["profile"])][r["method"]] = r
out = []
P = lambda s="": (print(s), out.append(s))
P("| network@load | telemetry | oracle | tuned (gain / p99) | learned, mean±SD over seeds (gain / p99) | ideal-trained (s0) | learned s0 raw / +guard (matched seed) | winner |")
P("|---|---|---|---|---|---|---|---|")
wins = defaultdict(int)
for (topo, prof), m in sorted(by.items(), key=lambda kv: (kv[0][0].split("@")[0], float(kv[0][0].split("@")[1]), ["ideal", "rich", "snmp", "degraded"].index(kv[0][1]))):
    seeds = [m[k] for k in m if k.startswith("gnn-ours-s") and "+" not in k]
    g = np.array([100 * s["gain"] for s in seeds]); p99 = np.array([s["regret_p99"] for s in seeds])
    t = m["tuned"]
    gt = 100 * t["gain"]
    best_l = g.mean()
    if abs(best_l - gt) < 1.0:
        win = "tie"
    else:
        win = "learned" if best_l > gt else "tuned"
    wins[(prof, win)] += 1
    orc = f"{100 * m['oracle']['gain']:.1f}" if "oracle" in m else "-"
    gi = f"{100 * m['gnn-ideal-s0']['gain']:.1f}" if "gnn-ideal-s0" in m else "-"
    lg = m.get("gnn-ours-s0+guard")
    raw0 = 100 * m["gnn-ours-s0"]["gain"]
    P(f"| {topo} | {prof} | {orc} | {gt:.1f} / {t['regret_p99']:+.3f} | {g.mean():.1f}±{g.std(ddof=1) if len(g) > 1 else 0:.1f} / {p99.mean():+.3f} "
      f"| {gi} | {raw0:.1f} / {100 * lg['gain']:.1f} (fb {lg.get('fallback_rate', 0):.2f}) | {win} |")
P("\nWinner counts (|gain difference| < 1 point = tie), over network x load cells:\n")
for prof in ("ideal", "rich", "snmp", "degraded"):
    P(f"- {prof}: learned {wins[(prof, 'learned')]}, tuned {wins[(prof, 'tuned')]}, tie {wins[(prof, 'tie')]}")
if "--md" in sys.argv:
    open(sys.argv[sys.argv.index("--md") + 1], "w", encoding="utf-8").write("\n".join(out) + "\n")
