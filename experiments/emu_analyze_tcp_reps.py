"""TCP emulation, 9 scenarios x 3 repeats (+ pre-declared variants): run-to-run noise and policy comparison on scenario means.
Repeats: runs/emu_tcp (rep 0), runs/emu_tcp_r1, runs/emu_tcp_r2. Variants (1 run each): runs/emu_tcp_var (guard L2 only), runs/emu_tcp_varh (learned, hyst 0.2)."""
import json
from pathlib import Path
import numpy as np
from scipy import stats
from stale_but_safe.emu import analyze as A, scenario as S
from stale_but_safe.emu.run_emulation import tag_of

NAMES = ["s1", "s2", "s3", "n14", "n15", "n16", "n17", "n18", "n19"]
ROOTS = {0: "runs/emu_tcp", 1: "runs/emu_tcp_r1", 2: "runs/emu_tcp_r2"}


def model_of(n, root):
    for r in (Path(root) / n / "model.txt", Path("runs/emu_tcp") / n / "model.txt"):
        if r.exists():
            return r.read_text().strip()
    return f"runs/ours_dfm_final_{n}.pt"


def metrics(root, n, spec, prof, skip=3):
    d = Path(root) / n
    try:
        em = A.load_emulation(d, tag_of(spec, prof), S.load(d / "scenario.json"))
    except (FileNotFoundError, KeyError):
        return None
    return float(np.nanmean(em["max_util"][skip:])), float(np.nanmean(em["loss"][skip:])), float(em["changed"].sum() / (len(em["changed"]) / 60))


POL = {"ECMP": lambda M: ("ecmp", "ideal"), "tuned": lambda M: ("tunedv2:degraded", "degraded"),
       "learned": lambda M: (f"model:{M}", "degraded"), "guard0.05": lambda M: (f"guard0.05@model:{M}", "degraded")}
res = {p: {n: [] for n in NAMES} for p in POL}
for rep, root in ROOTS.items():
    for n in NAMES:
        M = model_of(n, root)
        for p, f in POL.items():
            m = metrics(root, n, *f(M))
            if m: res[p][n].append(m)
VAR = {"guardL2-0.06+learned": ("runs/emu_tcp_var", lambda M: (f"guardL2-0.06@model:{M}", "degraded")),
       "learned, hyst 0.2": ("runs/emu_tcp_varh", lambda M: (f"model:{M}", "degraded"))}
for p, (root, f) in VAR.items():
    res[p] = {n: [] for n in NAMES}
    for n in NAMES:
        m = metrics(root, n, *f(model_of(n, root)))
        if m: res[p][n].append(m)
json.dump(res, open("experiments/final/emu_tcp_reps.json", "w"), indent=1)

full = [n for n in NAMES if all(len(res[p][n]) == 3 for p in POL)]
print(f"scenarios with 3 complete repeats: {len(full)}/9\n")
print("| policy | mean max-util | mean goodput shortfall | flow-mods/min | run-to-run SD of max-util (within scenario) | run-to-run SD of shortfall |\n|---|---|---|---|---|---|")
for p in res:
    ns = [n for n in NAMES if res[p][n]]
    if not ns: continue
    a = np.array([[x[0] for x in res[p][n]] for n in ns if len(res[p][n]) == 3] or [[np.nan] * 3])
    l = np.array([[x[1] for x in res[p][n]] for n in ns if len(res[p][n]) == 3] or [[np.nan] * 3])
    mu = np.mean([np.mean([x[0] for x in res[p][n]]) for n in ns]); ls = np.mean([np.mean([x[1] for x in res[p][n]]) for n in ns])
    fm = np.mean([np.mean([x[2] for x in res[p][n]]) for n in ns])
    print(f"| {p} (n={len(ns)}) | {mu:.3f} | {ls:.4f} | {fm:.0f} | {np.nanmean(a.std(axis=1, ddof=1)):.4f} | {np.nanmean(l.std(axis=1, ddof=1)):.4f} |")


def smean(p, n, k): return np.mean([x[k] for x in res[p][n]])


def paired(a, b, label):
    ns = [n for n in NAMES if res[a][n] and res[b][n]]
    for k, lab in [(0, "max-util"), (1, "goodput shortfall")]:
        d = np.array([smean(a, n, k) - smean(b, n, k) for n in ns])
        w = stats.wilcoxon(d).pvalue if np.any(d != 0) else float("nan")
        bs = np.random.default_rng(0).choice(d, (4000, len(d))).mean(1)
        print(f"{label}, {lab}: mean {d.mean():+.4f} [95% bootstrap {np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}], lower in {(d < 0).sum()}/{len(d)} scenarios, Wilcoxon p={w:.3f}")


print("\nPaired on scenario means (negative = first policy better):")
paired("learned", "tuned", "learned - tuned")
paired("learned", "ECMP", "learned - ECMP")
paired("tuned", "ECMP", "tuned - ECMP")
paired("guard0.05", "learned", "guard0.05 - learned")
paired("guardL2-0.06+learned", "learned", "guardL2 - learned")
paired("learned, hyst 0.2", "learned", "learned hyst0.2 - learned")
paired("learned, hyst 0.2", "tuned", "learned hyst0.2 - tuned")
