"""Tuning-variability check for the delay x polling grid.
For each cell, the heuristic is tuned K times with independent tuner seeds (random search, BUDGET trials, synthetic tuning graphs,
exactly as in delay_controlled.py but with more trials). Reported per cell: gain (%) of the seed whose *tuning* score was best
(selection never sees the evaluation data), and min/max gain across tuner seeds. Learned numbers are taken from table_delay_part2.md.
  python experiments/delay_tuning_repeat.py [workers] > experiments/final/table_delay_repeat.md ; raw: experiments/final/delay_repeat.json
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

K, BUDGET = 5, 80
CELLS = [(d, P) for d in (0, 3, 10) for P in (1, 5, 20)]
RAW = Path("experiments/final/delay_repeat.json")


def task(args):
    d, P, k = args
    from stale_but_safe.baselines.heuristics import TunedHeuristic, tune_heuristic
    from stale_but_safe.env import real
    from stale_but_safe.env.te_env import EnvCfg
    from stale_but_safe.env.telemetry import TelemetryCfg
    from stale_but_safe.env.topology import synthetic
    from stale_but_safe.env.zoo import load_split, zoo_topology
    from stale_but_safe.eval.compare import episode_stats
    from stale_but_safe.eval.tune import TUNE_TOPOS, score
    tele = TelemetryCfg(P, d, 0.0, 0.0)
    tt = [synthetic(a, n, s) for a, n, s in TUNE_TOPOS]
    mk = lambda: EnvCfg(load=0.8, horizon=60, telemetry=tele)
    best, sc = tune_heuristic(mk, lambda pol, m: score(pol, m, tt), budget=BUDGET, seed=100 + k)
    zoo = [t for t in (zoo_topology(n) for n in load_split("test_subset")) if t.n <= 25][:10]
    ab, atr = real.load("abilene"); ge, gtr = real.load("geant")
    out = {"d": d, "P": P, "k": k, "tune_score": sc, "gain": {}}
    for cname, topos, trace, ns in [("zoo", zoo, None, 3), ("abilene", [ab], atr, 6), ("geant", [ge], gtr, 6)]:
        g = []
        for t in topos:
            cfg = EnvCfg(load=0.8, horizon=100, telemetry=tele, trace=trace)
            for s in range(ns):
                r = episode_stats(t, cfg, TunedHeuristic(**best), 14000 + s)
                if r["ecmp_mu"] > 1e-9:
                    g.append(1 - r["mu"] / r["ecmp_mu"])
        out["gain"][cname] = 100 * float(np.mean(g))
    return out


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    done = json.load(open(RAW)) if RAW.exists() else []
    have = {(r["d"], r["P"], r["k"]) for r in done}
    todo = [(d, P, k) for d, P in CELLS for k in range(K) if (d, P, k) not in have]
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(task, todo):
            done.append(r); json.dump(done, open(RAW, "w"), indent=1)
    learned = {}
    for line in open("experiments/final/table_delay_part2.md").read().splitlines()[2:]:
        c = [x.strip() for x in line.strip("|").split("|")]
        learned[(c[0].split()[0].lower(), int(c[1]), int(c[2]))] = float(c[4])
    print(f"Heuristic re-tuned {K}x per cell (independent tuner seeds, {BUDGET} trials each). 'sel' = gain of the seed with best tuning score; "
          "range = min-max gain over tuner seeds; margin = learned - sel.\n")
    print("| case | d | P | heuristic sel | range over tuner seeds | learned | margin (learned - sel) | margin vs best seed |")
    print("|---|---|---|---|---|---|---|---|")
    for cn in ("zoo", "abilene", "geant"):
        for d, P in CELLS:
            rs = [r for r in done if (r["d"], r["P"]) == (d, P)]
            sel = min(rs, key=lambda r: r["tune_score"])["gain"][cn]
            gs = [r["gain"][cn] for r in rs]
            L = learned[(cn, d, P)]
            print(f"| {cn} | {d} | {P} | {sel:.1f} | {min(gs):.1f}-{max(gs):.1f} | {L:.1f} | {L - sel:+.1f} | {L - max(gs):+.1f} |")
