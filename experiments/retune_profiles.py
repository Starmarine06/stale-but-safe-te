"""Protocol-frozen re-tuning of the per-profile heuristic (stronger than configs/tuned_heuristic.yaml: 40 trials, 1 run).
Protocol (fixed before looking at the evaluation): for each profile in {ideal,rich,snmp,degraded}, K=5 independent tuner seeds
(100..104), BUDGET=80 random-search trials on the 4 synthetic tuning graphs (TUNE_TOPOS, horizon 60, load 0.8), same objective and search
space as the original. Each tuner seed's parameters are then evaluated on the 24 held-out Zoo test graphs exactly as final_eval.py (5 eval
seeds, horizon 100, load 0.8). Reported: mean over tuner seeds (expected performance), min/max, and the seed chosen by tuning score only.
  python experiments/retune_profiles.py [workers] -> experiments/final/retune_profiles.json (raw)
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

K, BUDGET = 5, 80
PROFILES_ = ("ideal", "rich", "snmp", "degraded")
RAW = Path("experiments/final/retune_profiles.json")


def task(args):
    prof, k = args
    from stale_but_safe.baselines.heuristics import TunedHeuristic, tune_heuristic
    from stale_but_safe.env.te_env import EnvCfg
    from stale_but_safe.env.telemetry import PROFILES
    from stale_but_safe.env.topology import synthetic
    from stale_but_safe.env.zoo import load_split, zoo_topology
    from stale_but_safe.eval.compare import ecmp_score, episode_stats, score
    from stale_but_safe.eval.tune import TUNE_TOPOS, score as tscore
    tele = PROFILES[prof]
    tt = [synthetic(a, n, s) for a, n, s in TUNE_TOPOS]
    best, sc = tune_heuristic(lambda: EnvCfg(load=0.8, horizon=60, telemetry=tele), lambda pol, m: tscore(pol, m, tt), budget=BUDGET, seed=100 + k)
    best = {a: float(b) for a, b in best.items()}
    cfg = EnvCfg(load=0.8, horizon=100, telemetry=tele)
    per = {}
    for name in load_split("test_subset"):
        t = zoo_topology(name)
        rows = [episode_stats(t, cfg, TunedHeuristic(**best), 5000 + i) for i in range(5)]
        reg = np.concatenate([r["regret"] for r in rows])
        per[t.name] = {"delta": float(np.mean([score(r) - ecmp_score(r) for r in rows])),
                       "gain": float(np.mean([1 - r["mu"] / r["ecmp_mu"] for r in rows])),
                       "regret_p99": float(np.quantile(reg, 0.99))}
    return {"profile": prof, "k": k, "tune_score": sc, "params": best, "per_topo": per}


if __name__ == "__main__":
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    done = json.load(open(RAW)) if RAW.exists() else []
    have = {(r["profile"], r["k"]) for r in done}
    todo = [(p, k) for p in PROFILES_ for k in range(K) if (p, k) not in have]
    with ProcessPoolExecutor(w) as ex:
        for r in ex.map(task, todo):
            done.append(r); json.dump(done, open(RAW, "w"))
