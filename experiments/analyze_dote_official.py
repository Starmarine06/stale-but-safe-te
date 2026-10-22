"""Table: DOTE (released architecture) vs re-tuned heuristic vs zero-shot GNN, per network, load and telemetry profile.
Gain = reduction of mean max-link-utilisation vs ECMP (percent); GNN is the mean of four training seeds."""
import json
import sys

import numpy as np

nets = sys.argv[1:] or ["abilene", "geant"]
print("| network | profile | ECMP gain | re-tuned heuristic | DOTE (ideal-trained) | DOTE (degraded-trained) | zero-shot GNN |")
print("|---|---|---|---|---|---|---|")
for n in nets:
    try:
        rows = json.load(open(f"experiments/final/dote_official_{n}.json"))["results"]
    except FileNotFoundError:
        continue
    for pn in ("ideal", "snmp", "degraded"):
        def g(m):
            v = [r["gain"] for r in rows if r["profile"] == pn and (r["method"] == m or (m == "gnn" and r["method"].startswith("gnn-ours")))]
            return 100 * float(np.mean(v)) if v else float("nan")
        print(f"| {n} | {pn} | {g('ecmp'):.1f} | {g('tuned'):.1f} | {g('dote-official-ideal'):.1f} | {g('dote-official-degraded'):.1f} | {g('gnn'):.1f} |")
