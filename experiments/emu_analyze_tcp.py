"""Packet-level (TCP) emulation, runs/emu_tcp/<seed>: goodput shortfall and max-util per policy, plus sim agreement."""
import json
from pathlib import Path
import numpy as np, yaml
from stale_but_safe.baselines.heuristics import ECMP, TunedHeuristic
from stale_but_safe.emu import analyze as A, scenario as S
from stale_but_safe.guard.guard import Guard, GuardCfg
from stale_but_safe.models.learned import LearnedPolicy, load

v2 = yaml.safe_load(open("configs/tuned_heuristic_v2.yaml"))


def make_policy(spec, profile):
    eps = None
    if spec.startswith("guard"):
        head, spec = spec.split("@", 1)
        eps = float(head[5:] or 0.05)
    if spec == "ecmp":
        pol = ECMP()
    elif spec.startswith("tunedv2:"):
        pol = TunedHeuristic(**v2[spec.split(":", 1)[1]]["params"])
    else:
        m, c = load(spec.split(":", 1)[1]); pol = LearnedPolicy(m, c, "gnn")
    return Guard(pol, GuardCfg(eps=eps, layers=(True, True, True))) if eps is not None else pol


rows = []
for d in sorted(p for p in Path("runs/emu_tcp").iterdir() if p.is_dir() and any(p.glob("tuned*/done.json"))):
    s = d.name
    mf = d / "model.txt"
    M = "model:" + (mf.read_text().strip() if mf.exists() else f"runs/ours_dfm_final_{s}.pt")
    runs = [("ecmp", "ideal"), ("tunedv2:degraded", "degraded"), (M, "degraded"), (f"guard0.05@{M}", "degraded")]
    for r in A.compare(d, S.load(d / "scenario.json"), runs, make_policy):
        r["seed"] = s; rows.append(r)
json.dump(rows, open("experiments/final/emu_tcp_rows.json", "w"), indent=1, default=float)
names = {"ecmp": "ECMP", "tunedv2": "tuned heuristic [degraded]", "model": "learned [degraded]", "guard": "guard(0.05)+learned [degraded]"}
key = lambda r: names["guard" if r["spec"].startswith("guard") else r["spec"].split(":")[0]]
print(f"TCP emulation, {len(set(r['seed'] for r in rows))} seeds\n")
print("| policy | emu max-util | sim max-util | goodput shortfall (emu) | sim loss | flow-mods/min | per-seed emu max-util |\n|---|---|---|---|---|---|---|")
for n in names.values():
    rs = [r for r in rows if key(r) == n]
    if not rs: continue
    f = lambda k: np.mean([r[k] for r in rs])
    print(f"| {n} | {f('emu_mu'):.3f} | {f('sim_mu'):.3f} | {f('emu_loss'):.4f} | {f('sim_loss'):.4f} | {f('flowmods_per_min'):.1f} | "
          + ", ".join(f"{r['emu_mu']:.3f}" for r in rs) + " |")
from scipy import stats
def per(n, k): return {r["seed"]: r[k] for r in rows if key(r) == n}
L, T, E, G = per(names["model"], "emu_mu"), per(names["tunedv2"], "emu_mu"), per(names["ecmp"], "emu_mu"), per(names["guard"], "emu_mu")
Ll, Tl = per(names["model"], "emu_loss"), per(names["tunedv2"], "emu_loss")
ks = sorted(set(L) & set(T))
for lab, a, b in [("max-util", L, T), ("goodput shortfall", Ll, Tl)]:
    dlt = np.array([a[k] - b[k] for k in ks])
    w = stats.wilcoxon(dlt).pvalue if np.any(dlt != 0) else float("nan")
    print(f"\nlearned minus tuned,{lab}, n={len(ks)} scenarios: mean {dlt.mean():+.4f}, learned lower in {(dlt < 0).sum()}/{len(ks)}, Wilcoxon p={w:.3f}")
dg = np.array([G[k] - L[k] for k in sorted(set(G) & set(L))])
print(f"guard minus unguarded learned, max-util: mean {dg.mean():+.4f}, guard lower in {(dg < 0).sum()}/{len(dg)}")
e = np.array([r["emu_mu"] for r in rows]); sm = np.array([r["sim_mu"] for r in rows])
print(f"\nn={len(rows)} runs: corr(emu, sim max-util) = {np.corrcoef(e, sm)[0,1]:.2f}; mean |emu-sim| = {np.abs(e-sm).mean():.3f}")
