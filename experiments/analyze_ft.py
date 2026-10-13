"""Held-out-window comparison on real networks: tuned, static LP, DOTE-style per-network, zero-shot GNN, fine-tuned GNN."""
import json
from collections import defaultdict
F = "experiments/final/"
by = defaultdict(dict)
for net in ("abilene", "geant"):
    for f in (f"dote_{net}.json", f"static_{net}.json", f"ft_{net}.json"):
        for r in json.load(open(F + f))["results"]:
            if r["method"] in ("tuned", "dote-ideal", "dote-degraded", "static-lp", "gnn-zeroshot", "gnn-finetuned", "gnn-ours-s0"):
                by[(r["topo"], r["profile"])][r["method"]] = r
cols = [("tuned", "tuned"), ("static-lp", "static LP"), ("dote-ideal", "DOTE-style (ideal-trained)"), ("dote-degraded", "DOTE-style (degraded-trained)"),
        ("gnn-zeroshot", "GNN zero-shot"), ("gnn-finetuned", "GNN fine-tuned on history")]
print("| network (held-out windows) @load | telemetry | " + " | ".join(c[1] for c in cols) + " |")
print("|---|---|" + "---|" * len(cols))
for (topo, prof), m in sorted(by.items(), key=lambda kv: (kv[0][0], ["ideal", "snmp", "degraded"].index(kv[0][1]))):
    cells = [f"{100 * m[k]['gain']:.1f} / {m[k]['regret_p99']:+.3f}" if k in m else "-" for k, _ in cols]
    print(f"| {topo} | {prof} | " + " | ".join(cells) + " |")
print("\n(gain % vs ECMP / p99 regret; seed-0 model for the GNN columns; 10 held-out windows per cell)")
