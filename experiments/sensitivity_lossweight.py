"""Does the conclusion depend on the loss weight (5) in score = maxutil/ecmp + w*loss?
dScore_w = -gain + w*(loss - ecmp_loss), recomputed per topology from the stored per-method means."""
import json
from collections import defaultdict
import numpy as np

rows = json.load(open("experiments/final/results_ours.json"))["results"]
by = defaultdict(dict)
for r in rows:
    by[(r["profile"], r["topo"])][r["method"]] = r
print("| telemetry | w | tuned dScore | learned dScore | learned beats tuned on |")
print("|---|---|---|---|---|")
for prof in ("ideal", "rich", "snmp", "degraded"):
    for w in (0, 1, 5, 10, 20):
        d_t, d_l = [], []
        for (p, topo), m in by.items():
            if p != prof or "gnn-ours" not in m or "tuned" not in m or "ecmp" not in m:
                continue
            f = lambda k: -m[k]["gain"] + w * (m[k]["loss"] - m["ecmp"]["loss"])
            d_t.append(f("tuned")); d_l.append(f("gnn-ours"))
        d_t, d_l = np.array(d_t), np.array(d_l)
        print(f"| {prof} | {w} | {d_t.mean():+.3f} | {d_l.mean():+.3f} | {int((d_l < d_t).sum())} of {len(d_t)} |")
