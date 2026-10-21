import json, numpy as np
from pathlib import Path
from scipy import stats
from stale_but_safe.emu import analyze as A, scenario as S
from stale_but_safe.emu.run_emulation import tag_of
N = ["s1","s2","s3","n14","n15","n16","n17","n18","n19"]
def mo(n):
    p = Path("runs/emu_tcp")/n/"model.txt"
    return p.read_text().strip() if p.exists() else f"runs/ours_dfm_final_{n}.pt"
def m(root, n, spec, prof="degraded"):
    d = Path(root)/n
    em = A.load_emulation(d, tag_of(spec, prof), S.load(d/"scenario.json"))
    fb = np.array([r["fallback"] for r in em["rows"]])
    return (float(np.nanmean(em["max_util"][3:])), float(np.nanmean(em["loss"][3:])),
            float(em["changed"].sum()/(len(em["changed"])/60)), float(fb.mean()), int((np.diff(fb)!=0).sum()))
P = {"learned h0.2": ("runs/emu_tcp_varh", lambda M: f"model:{M}"),
     "guard0.05 h0.2": ("runs/emu_tcp_gh", lambda M: f"guard0.05@model:{M}"),
     "guardL2-0.06 h0.2": ("runs/emu_tcp_gh", lambda M: f"guardL2-0.06@model:{M}"),
     "guard0.05 h0.05": ("runs/emu_tcp", lambda M: f"guard0.05@model:{M}"),
     "learned h0.05": ("runs/emu_tcp", lambda M: f"model:{M}"),
     "tuned h0.05": ("runs/emu_tcp", lambda M: "tunedv2:degraded")}
R = {}
for p,(root,f) in P.items():
    R[p] = {}
    for n in N:
        try: R[p][n] = m(root, n, f(mo(n)))
        except Exception as e: R[p][n] = None
print("| policy | n | max-util | shortfall | flow-mods/min | fallback% | transitions |\n|---|---|---|---|---|---|---|")
for p in P:
    v = [R[p][n] for n in N if R[p][n]]
    print(f"| {p} | {len(v)} | {np.mean([x[0] for x in v]):.3f} | {np.mean([x[1] for x in v]):.4f} | {np.mean([x[2] for x in v]):.0f} | {100*np.mean([x[3] for x in v]):.0f} | {np.mean([x[4] for x in v]):.1f} |")
def pair(a,b,i,label):
    ns=[n for n in N if R[a][n] and R[b][n]]; d=np.array([R[a][n][i]-R[b][n][i] for n in ns])
    bs=[np.mean(np.random.default_rng(s).choice(d,len(d))) for s in range(4000)]
    print(f"{label}: {a} - {b}: mean {d.mean():+.4f} [{np.quantile(bs,.025):+.4f},{np.quantile(bs,.975):+.4f}] lower in {(d<0).sum()}/{len(d)} p={stats.wilcoxon(d).pvalue:.3f}")
for a,b in [("guard0.05 h0.2","learned h0.2"),("guardL2-0.06 h0.2","learned h0.2"),("guard0.05 h0.2","tuned h0.05"),("guard0.05 h0.2","guard0.05 h0.05")]:
    pair(a,b,0,"max-util"); pair(a,b,1,"shortfall")
