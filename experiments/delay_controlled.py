"""Controlled staleness experiment.
  part 1: delay d in {0,1,3,5,10}; tuned heuristic with FROZEN parameters (tuned for ideal / for degraded telemetry) vs
          parameters RE-TUNED at each d (40-trial random search on 4 synthetic graphs), vs the learned policy (mean of 4 training seeds).
  part 2: interaction d x P (polling period); heuristic re-tuned per cell; learned mean of 4 seeds.
Everything else ideal. Evaluated on 10 Zoo graphs (n<=25), Abilene and GEANT traces at load 0.8.
  python experiments/delay_controlled.py 1|2   -> experiments/final/table_delay_part{1,2}.md, retuned params in experiments/final/retuned_part{1,2}.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from stale_but_safe.baselines.heuristics import TunedHeuristic, tune_heuristic
from stale_but_safe.env import real
from stale_but_safe.env.te_env import EnvCfg
from stale_but_safe.env.telemetry import TelemetryCfg
from stale_but_safe.env.topology import synthetic
from stale_but_safe.env.zoo import load_split, zoo_topology
from stale_but_safe.eval.compare import episode_stats
from stale_but_safe.eval.tune import TUNE_TOPOS, score
from stale_but_safe.models.learned import LearnedPolicy, load

part = int(sys.argv[1])
tuned = yaml.safe_load(open("configs/tuned_heuristic.yaml"))
models = [load(p) for p in ["runs/ours_dfm_final.pt", "runs/ours_dfm_final_s1.pt", "runs/ours_dfm_final_s2.pt", "runs/ours_dfm_final_s3.pt"]]
zoo = [t for t in (zoo_topology(n) for n in load_split("test_subset")) if t.n <= 25][:10]
ab, atr = real.load("abilene"); ge, gtr = real.load("geant")
cases = [("Zoo (10 graphs)", zoo, None, 3), ("Abilene real", [ab], atr, 6), ("GEANT real", [ge], gtr, 6)]
tune_topos = [synthetic(k, n, s) for k, n, s in TUNE_TOPOS]
cache_f = Path(f"experiments/final/retuned_part{part}.json")
cache = json.load(open(cache_f)) if cache_f.exists() else {}


def retune(tele: TelemetryCfg, key: str):
    if key not in cache:
        mk = lambda: EnvCfg(load=0.8, horizon=60, telemetry=tele)
        best, sc = tune_heuristic(mk, lambda pol, mk_: score(pol, mk_, tune_topos), budget=40, seed=1)
        cache[key] = {k: float(v) for k, v in best.items()}
        json.dump(cache, open(cache_f, "w"), indent=1)
    return cache[key]


def gain(pols, tele, topos, trace, nseeds):
    g = []
    for pol in pols:
        for t in topos:
            cfg = EnvCfg(load=0.8, horizon=100, telemetry=tele, trace=trace)
            for s in range(nseeds):
                r = episode_stats(t, cfg, pol, 14000 + s)
                if r["ecmp_mu"] > 1e-9:          # real traces contain zero-traffic gaps; skip those windows
                    g.append(1 - r["mu"] / r["ecmp_mu"])
    return 100 * float(np.mean(g))


if part == 1:
    print("| case | delay d | tuned: frozen (clean-tuned) | tuned: frozen (degraded-tuned) | tuned: re-tuned at this d | learned (mean of 4 seeds) |")
    print("|---|---|---|---|---|---|")
    for d in (0, 1, 3, 5, 10):
        tele = TelemetryCfg(1, d, 0.0, 0.0)
        rp = retune(tele, f"d{d}")
        for cname, topos, trace, ns in cases:
            row = [gain([TunedHeuristic(**tuned["ideal"]["params"])], tele, topos, trace, ns),
                   gain([TunedHeuristic(**tuned["degraded"]["params"])], tele, topos, trace, ns),
                   gain([TunedHeuristic(**rp)], tele, topos, trace, ns),
                   gain([LearnedPolicy(m, c, "g") for m, c in models], tele, topos, trace, ns)]
            print(f"| {cname} | {d} | " + " | ".join(f"{x:.1f}" for x in row) + " |", flush=True)
else:
    print("| case | delay d | polling P | tuned (re-tuned in this cell) | learned (mean of 4 seeds) | learned - tuned |")
    print("|---|---|---|---|---|---|")
    for d in (0, 3, 10):
        for P in (1, 5, 20):
            tele = TelemetryCfg(P, d, 0.0, 0.0)
            rp = retune(tele, f"d{d}_P{P}")
            for cname, topos, trace, ns in cases:
                a = gain([TunedHeuristic(**rp)], tele, topos, trace, ns)
                b = gain([LearnedPolicy(m, c, "g") for m, c in models], tele, topos, trace, ns)
                print(f"| {cname} | {d} | {P} | {a:.1f} | {b:.1f} | {b - a:+.1f} |", flush=True)
