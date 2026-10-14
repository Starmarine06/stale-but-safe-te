"""Guard and stress comparisons against the re-tuned heuristic (configs/tuned_heuristic_v2.yaml).
  python experiments/analyze_v2_guard_stress.py > experiments/final/table_v2_guard_stress.md"""
import json
from collections import defaultdict
import numpy as np
F = "experiments/final/"
rng = np.random.default_rng(0)
def load(paths, methods):
    d = defaultdict(dict)
    for p in paths:
        for r in json.load(open(p))["results"]:
            if r["method"] in methods:
                d[(r["profile"], r["method"])][r["topo"]] = r
    return d
def line(d, prof, m, common, ref=None):
    x = np.array([d[(prof, m)][t]["delta"] for t in common]); p = np.mean([d[(prof, m)][t]["regret_p99"] for t in common])
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(3000)]; lo, hi = np.percentile(b, [2.5, 97.5])
    fb = [d[(prof, m)][t].get("fallback_rate") for t in common]; fbs = f"{np.mean([f for f in fb if f is not None]):.2f}" if any(f is not None for f in fb) else "-"
    return f"| {prof} | {m} | {x.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | {p:.3f} | {fbs} |"
print("### Zoo, guard L2-only eps=-0.06, 24 graphs. Heuristic = re-tuned (best tuning score). dScore vs ECMP, lower better; 95% CI over graphs\n")
print("| profile | method | dScore | p99 regret | guard fallback rate |\n|---|---|---|---|---|")
d = load([F + "results_guardL2.json"], {"gnn-ours", "gnn-ours+guard"}); d.update(load([F + "results_v2_zoo.json"], {"tuned", "tuned+guard"}))
for prof in ("ideal", "rich", "snmp", "degraded"):
    common = sorted(set.intersection(*[set(d[(prof, m)]) for m in ("gnn-ours", "gnn-ours+guard", "tuned", "tuned+guard")]))
    for m in ("gnn-ours", "gnn-ours+guard", "tuned", "tuned+guard"):
        print(line(d, prof, m, common))
print("\n### Stress scenarios (12 graphs, original eval protocol), learned vs re-tuned heuristic; diff = heuristic - learned (positive = learned better)\n")
print("| scenario | profile | learned dScore | re-tuned heuristic dScore | diff [95% CI over graphs] | learned W/T/L | p99 regret learned / heuristic | original-heuristic diff |\n|---|---|---|---|---|---|---|---|")
for sc in ("fail", "shift", "flash", "hotspot"):
    try:
        old = load([F + f"stress_{sc}.json"], {"gnn-ours", "tuned"}); new = load([F + f"stress_v2_{sc}.json"], {"tuned"})
    except FileNotFoundError:
        continue
    for prof in ("snmp", "degraded"):
        common = sorted(set(old[(prof, "gnn-ours")]) & set(new[(prof, "tuned")]))
        if not common: continue
        L = np.array([old[(prof, "gnn-ours")][t]["delta"] for t in common]); H = np.array([new[(prof, "tuned")][t]["delta"] for t in common])
        O = np.array([old[(prof, "tuned")][t]["delta"] for t in common]); df = H - L
        b = [df[rng.integers(0, len(df), len(df))].mean() for _ in range(3000)]; lo, hi = np.percentile(b, [2.5, 97.5])
        w = int((df > 0.002).sum()); l = int((df < -0.002).sum())
        pl = np.mean([old[(prof, "gnn-ours")][t]["regret_p99"] for t in common]); ph = np.mean([new[(prof, "tuned")][t]["regret_p99"] for t in common])
        print(f"| {sc} | {prof} | {L.mean():+.3f} | {H.mean():+.3f} | {df.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | {w}/{len(df)-w-l}/{l} | {pl:.3f} / {ph:.3f} | {(O-L).mean():+.3f} |")
