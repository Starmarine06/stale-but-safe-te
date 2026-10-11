"""Tune the heuristic baseline per telemetry profile with a fixed budget."""
from __future__ import annotations

import numpy as np
import yaml

from ..baselines.heuristics import TunedHeuristic, tune_heuristic
from ..env.te_env import EnvCfg
from ..env.telemetry import PROFILES
from ..env.topology import synthetic
from .runner import evaluate

TUNE_TOPOS = [("ba", 12, 9001), ("er", 14, 9002), ("geo", 14, 9003), ("ba", 18, 9004)]
TUNE_SEEDS = (9100, 9101, 9102)


def score(policy, make_cfg, topos):
    ratios = []
    for t in topos:
        r = evaluate(t, make_cfg(), policy, TUNE_SEEDS)
        ratios.append(r["mean_max_util"] / r["mean_ecmp_max_util"] + 5 * r["mean_loss"])
    return float(np.mean(ratios))


def tune_all(budget=40, load=0.8, horizon=60, out="configs/tuned_heuristic.yaml"):
    topos = [synthetic(k, n, s) for k, n, s in TUNE_TOPOS]
    res = {}
    for name, tele in PROFILES.items():
        mk = lambda tele=tele: EnvCfg(load=load, horizon=horizon, telemetry=tele)
        best, sc = tune_heuristic(mk, lambda pol, mk_: score(pol, mk_, topos), budget=budget)
        best = {k: float(v) for k, v in best.items()}
        res[name] = {"params": best, "score": sc, "budget": budget}
        print(name, round(sc, 3), best, flush=True)
    with open(out, "w") as f:
        yaml.safe_dump(res, f)
    return res


if __name__ == "__main__":
    tune_all()
