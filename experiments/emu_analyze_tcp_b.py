"""TCP emulation, batch 2 (18 pre-declared scenarios, docs/tcp_batch2_design.md) pooled with batch 1 (9 scenarios): learned vs tuned vs ECMP.
Primary outcome (declared before the runs): paired max-utilisation difference learned - tuned over scenario means (mean of the available
repeats, intervals 3 onward), bootstrap 95% interval and Wilcoxon. Also: shortfall difference, results by traffic pattern (descriptive),
and agreement with the simulator replay (experiments/final/sim_replay_*.json) of the same scenarios.
  python experiments/emu_analyze_tcp_b.py [n_repeats]   (default: all repeats present; pass 1 for the first run only)"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy import stats
from stale_but_safe.emu import analyze as A, scenario as S
from stale_but_safe.emu.run_emulation import tag_of

B1 = {"roots": ["runs/emu_tcp", "runs/emu_tcp_r1", "runs/emu_tcp_r2"], "ids": ["s1", "s2", "s3", "n14", "n15", "n16", "n17", "n18", "n19"]}
B2 = {"roots": ["runs/emu_tcp_b", "runs/emu_tcp_b_r1", "runs/emu_tcp_b_r2"], "ids": [f"b{i:02d}" for i in range(1, 19)]}
PATTERN = {**{n: "batch 1 (gravity + bursts, load 0.8)" for n in B1["ids"]},
           **{f"b{i:02d}": ("gravity, load 0.8" if i <= 3 else "hotspots" if i <= 6 else "demand shift" if i <= 9 else "flash crowd" if i <= 12
                           else "gravity, load 0.6" if i in (13, 14) else "gravity, load 1.0" if i in (15, 16) else "Abilene hotspots/flash") for i in range(1, 19)}}


NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def model_of(root, n):
    for r in (Path(root) / n / "model.txt", Path(B1["roots"][0]) / n / "model.txt", Path(B2["roots"][0]) / n / "model.txt"):
        if r.exists():
            return r.read_text().strip()
    return f"runs/ours_dfm_final_{n}.pt"


def metric(root, n, spec, prof, skip=3):
    d = Path(root) / n
    try:
        em = A.load_emulation(d, tag_of(spec, prof), S.load(d / "scenario.json"))
    except (FileNotFoundError, KeyError):
        return None
    return float(np.nanmean(em["max_util"][skip:])), float(np.nanmean(em["loss"][skip:]))


def collect(batch):
    out = {}
    for n in batch["ids"]:
        row = {}
        for pol, f in (("ecmp", lambda M: ("ecmp", "ideal")), ("tuned", lambda M: ("tunedv2:degraded", "degraded")), ("learned", lambda M: (f"model:{M}", "degraded"))):
            ms = [m for r in batch["roots"][:NREP] if (m := metric(r, n, *f(model_of(r, n)))) is not None]
            row[pol] = (np.mean([m[0] for m in ms]), np.mean([m[1] for m in ms]), len(ms)) if ms else None
        if all(row.values()):
            out[n] = row
    return out


def boot(d, k=4000):
    rng = np.random.default_rng(0)
    b = [np.mean(rng.choice(d, len(d))) for _ in range(k)]
    return np.quantile(b, [.025, .975])


def report(label, res, sim):
    ids = list(res)
    print(f"\n### {label}: {len(ids)} scenarios (repeats per scenario: {sorted(set(res[n]['learned'][2] for n in ids))})")
    for a, b, i, name in (("learned", "tuned", 0, "max-util"), ("learned", "ecmp", 0, "max-util"), ("tuned", "ecmp", 0, "max-util"), ("learned", "tuned", 1, "goodput shortfall")):
        d = np.array([res[n][a][i] - res[n][b][i] for n in ids]); lo, hi = boot(d)
        print(f"{a} - {b}, {name}: mean {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}], lower in {(d < 0).sum()}/{len(d)}, Wilcoxon p = {stats.wilcoxon(d).pvalue:.3f}")
    if sim:
        emu_d = np.array([res[n]["learned"][0] - res[n]["tuned"][0] for n in ids]); sim_d = np.array([sim[n] for n in ids])
        print(f"simulator replay predicts learned - tuned {sim_d.mean():+.4f}; emulation measures {emu_d.mean():+.4f}; per-scenario Pearson r = {np.corrcoef(emu_d, sim_d)[0, 1]:+.2f}, "
              f"sign agreement {(np.sign(emu_d) == np.sign(sim_d)).sum()}/{len(ids)}")


def sim_diffs():
    sim = {}
    for f in ("experiments/final/sim_replay_emu_tcp.json", "experiments/final/sim_replay_emu_tcp_b.json"):
        if Path(f).exists():
            j = json.load(open(f))
            for n, l, t in zip(j["ids"], j["learned"], j["tuned"]):
                sim[n] = l - t
    return sim


if __name__ == "__main__":
    r1, r2 = collect(B1), collect(B2); sim = sim_diffs()
    report("Batch 1", r1, sim); report("Batch 2 (pre-declared, primary)", r2, sim); report("Batch 1 + batch 2 (all scenarios)", {**r1, **r2}, sim)
    print("\n| pattern | scenarios | learned - tuned max-util (mean over scenarios) | simulator replay | learned - ECMP |\n|---|---|---|---|---|")
    allr = {**r1, **r2}
    for pat in dict.fromkeys(PATTERN.values()):
        ids = [n for n in allr if PATTERN[n] == pat]
        if ids:
            print(f"| {pat} | {len(ids)} | {np.mean([allr[n]['learned'][0] - allr[n]['tuned'][0] for n in ids]):+.4f} | {np.mean([sim[n] for n in ids if n in sim]):+.4f} | "
                  f"{np.mean([allr[n]['learned'][0] - allr[n]['ecmp'][0] for n in ids]):+.4f} |")
