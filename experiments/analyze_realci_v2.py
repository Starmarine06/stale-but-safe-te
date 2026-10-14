"""Paired bootstrap intervals over trace windows: learned (mean of 4 training seeds per window) minus tuned, per cell."""
import glob, json
import numpy as np
rng = np.random.default_rng(0)
print("| network | load | telemetry | tuned gain % | learned gain % | learned - tuned, pts (95% paired CI over windows) | learned better in windows | verdict |")
print("|---|---|---|---|---|---|---|---|")
cnt = {"learned": 0, "tuned": 0, "tie": 0}
by_prof = {}
for f in sorted(glob.glob("experiments/final/realci_v2_*.json")):
    o = json.load(open(f))
    for key, cell in o["cells"].items():
        ld, pn = key.split("|")
        t = np.array(cell["tuned"]) * 100
        l = np.mean([cell[k] for k in cell if k.startswith("learned")], 0) * 100
        ok = np.isfinite(t) & np.isfinite(l)
        t, l = t[ok], l[ok]
        d = l - t
        bs = [rng.choice(d, len(d)).mean() for _ in range(4000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        v = "learned" if lo > 0 else "tuned" if hi < 0 else "tie"
        cnt[v] += 1; by_prof.setdefault(pn, {"learned": 0, "tuned": 0, "tie": 0})[v] += 1
        print(f"| {o['network']} | {ld} | {pn} | {t.mean():.1f} | {l.mean():.1f} | {d.mean():+.1f} [{lo:+.1f}, {hi:+.1f}] | {int((d > 0).sum())}/{len(d)} | {v} |")
print("\nCells where the paired 95% interval excludes 0:", cnt)
for pn in ("ideal", "rich", "snmp", "degraded"):
    if pn in by_prof: print(f"  {pn}: {by_prof[pn]}")
