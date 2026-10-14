"""More Abilene windows for the delay x polling grid (settles whether the sign of the Abilene result depends on the window sample).
Uses the heuristic parameters already selected in delay_ci.json (no re-tuning, no real data in tuning) and 48 new windows (seeds 16000+).
  python experiments/abilene_more_windows.py [workers] > experiments/final/table_abilene_more.md ; raw: experiments/final/abilene_more.json"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

NW = 48
CELLS = [(d, P) for d in (0, 3, 10) for P in (1, 5, 20)]
RAW = Path("experiments/final/abilene_more.json")


def task(args):
    d, P, params = args
    from stale_but_safe.baselines.heuristics import TunedHeuristic
    from stale_but_safe.env import real
    from stale_but_safe.env.te_env import EnvCfg
    from stale_but_safe.env.telemetry import TelemetryCfg
    from stale_but_safe.eval.compare import episode_stats
    from stale_but_safe.models.learned import LearnedPolicy, load
    models = [load(p) for p in ["runs/ours_dfm_final.pt", "runs/ours_dfm_final_s1.pt", "runs/ours_dfm_final_s2.pt", "runs/ours_dfm_final_s3.pt"]]
    topo, tr = real.load("abilene")
    cfg = EnvCfg(load=0.8, horizon=100, telemetry=TelemetryCfg(P, d, 0.0, 0.0), trace=tr)
    H, L = [], []
    for s in range(NW):
        r = episode_stats(topo, cfg, TunedHeuristic(**params), 16000 + s)
        if r["ecmp_mu"] <= 1e-9:
            continue
        H.append(1 - r["mu"] / r["ecmp_mu"])
        L.append(np.mean([1 - episode_stats(topo, cfg, LearnedPolicy(m, c, "g"), 16000 + s)["mu"] / r["ecmp_mu"] for m, c in models]))
    return {"d": d, "P": P, "heur": H, "learned": L}


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    prev = {(r["d"], r["P"]): r["params"] for r in json.load(open("experiments/final/delay_ci.json"))}
    done = json.load(open(RAW)) if RAW.exists() else []
    have = {(r["d"], r["P"]) for r in done}
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(task, [(d, P, prev[(d, P)]) for d, P in CELLS if (d, P) not in have]):
            done.append(r)
            json.dump(done, open(RAW, "w"))
    rng = np.random.default_rng(0)
    print(f"Abilene, load 0.8, {NW} new windows per cell (zero-traffic windows skipped); learned minus heuristic gain in points, 95% bootstrap CI over windows.\n")
    print("| d | P | heuristic % | learned % | difference [95% CI] | verdict | windows |\n|---|---|---|---|---|---|---|")
    for d, P in CELLS:
        r = next(x for x in done if (x["d"], x["P"]) == (d, P))
        h, l = 100 * np.array(r["heur"]), 100 * np.array(r["learned"])
        df = l - h
        bs = [df[rng.integers(0, len(df), len(df))].mean() for _ in range(4000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        v = "L" if lo > 0 else "H" if hi < 0 else "="
        print(f"| {d} | {P} | {h.mean():.1f} | {l.mean():.1f} | {df.mean():+.1f} [{lo:+.1f}, {hi:+.1f}] | {v} | {len(df)} |")
