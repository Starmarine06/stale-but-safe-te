"""Window-level paired CIs for the delay x polling grid (real traces, load 0.8).
Heuristic = tuner seed with best tuning score from delay_repeat.json (re-run deterministically, seed 100+k; tuning uses only the 4
synthetic graphs, never real data). Evaluation windows (seeds 15000+) are fresh: not used in tuning or in earlier tables.
Per window: gain of heuristic and mean gain of the 4 learned training seeds; paired difference; bootstrap 95% CI over windows.
  python experiments/delay_grid_ci.py [workers] > experiments/final/table_delay_ci.md
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

NW = 16
CELLS = [(d, P) for d in (0, 3, 10) for P in (1, 5, 20)]
RAW = Path("experiments/final/delay_ci.json")


def task(args):
    d, P, k = args
    from stale_but_safe.baselines.heuristics import TunedHeuristic, tune_heuristic
    from stale_but_safe.env import real
    from stale_but_safe.env.te_env import EnvCfg
    from stale_but_safe.env.telemetry import TelemetryCfg
    from stale_but_safe.env.topology import synthetic
    from stale_but_safe.eval.compare import episode_stats
    from stale_but_safe.eval.tune import TUNE_TOPOS, score
    from stale_but_safe.models.learned import LearnedPolicy, load
    tele = TelemetryCfg(P, d, 0.0, 0.0)
    tt = [synthetic(a, n, s) for a, n, s in TUNE_TOPOS]
    best, sc = tune_heuristic(lambda: EnvCfg(load=0.8, horizon=60, telemetry=tele), lambda pol, m: score(pol, m, tt), budget=80, seed=100 + k)
    models = [load(p) for p in ["runs/ours_dfm_final.pt", "runs/ours_dfm_final_s1.pt", "runs/ours_dfm_final_s2.pt", "runs/ours_dfm_final_s3.pt"]]
    out = {"d": d, "P": P, "k": k, "tune_score": sc, "params": {a: float(b) for a, b in best.items()}, "net": {}}
    for name in ("abilene", "geant"):
        topo, tr = real.load(name)
        cfg = EnvCfg(load=0.8, horizon=100, telemetry=tele, trace=tr)
        H, L = [], []
        for s in range(NW):
            r = episode_stats(topo, cfg, TunedHeuristic(**best), 15000 + s)
            if r["ecmp_mu"] <= 1e-9:
                continue
            H.append(1 - r["mu"] / r["ecmp_mu"])
            L.append(np.mean([1 - episode_stats(topo, cfg, LearnedPolicy(m, c, "g"), 15000 + s)["mu"] / r["ecmp_mu"] for m, c in models]))
        out["net"][name] = {"heur": H, "learned": L}
    return out


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    rep = json.load(open("experiments/final/delay_repeat.json"))
    sel = {c: min([r for r in rep if (r["d"], r["P"]) == c], key=lambda r: r["tune_score"])["k"] for c in CELLS}
    done = json.load(open(RAW)) if RAW.exists() else []
    have = {(r["d"], r["P"]) for r in done}
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(task, [(d, P, sel[(d, P)]) for d, P in CELLS if (d, P) not in have]):
            done.append(r); json.dump(done, open(RAW, "w"), indent=1)
    rng = np.random.default_rng(0)
    print(f"Paired window-level comparison, {NW} fresh windows per cell (zero-traffic windows skipped), load 0.8. diff = learned - heuristic gain (points);\n"
          "95% bootstrap CI over windows. Heuristic = tuner seed with best tuning score. Verdict: L = CI above 0, H = CI below 0, = otherwise.\n")
    print("| network | d | P | heuristic % | learned % | diff [95% CI] | verdict | windows |\n|---|---|---|---|---|---|---|---|")
    for n in ("abilene", "geant"):
        for d, P in CELLS:
            r = next(x for x in done if (x["d"], x["P"]) == (d, P))["net"][n]
            h, l = 100 * np.array(r["heur"]), 100 * np.array(r["learned"]); df = l - h
            bs = [df[rng.integers(0, len(df), len(df))].mean() for _ in range(4000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            v = "L" if lo > 0 else "H" if hi < 0 else "="
            print(f"| {n} | {d} | {P} | {h.mean():.1f} | {l.mean():.1f} | {df.mean():+.1f} [{lo:+.1f}, {hi:+.1f}] | {v} | {len(df)} |")
