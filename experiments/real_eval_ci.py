"""Per-window real-trace evaluation (tuned vs learned, 4 training seeds) so that paired intervals over windows can be reported.
  python experiments/real_eval_ci.py abilene   -> experiments/final/realci_abilene.json
Each evaluation seed = one random trace window of 100 intervals. Stores per-window gain (fraction of ECMP max-util removed)."""
import json
import sys
import time

import numpy as np
import yaml

from stale_but_safe.baselines.heuristics import TunedHeuristic
from stale_but_safe.env import real
from stale_but_safe.env.te_env import EnvCfg
from stale_but_safe.env.telemetry import PROFILES
from stale_but_safe.eval.compare import episode_stats
from stale_but_safe.models.learned import LearnedPolicy, load

net = sys.argv[1]
W = int(sys.argv[2]) if len(sys.argv) > 2 else 12
tuned = yaml.safe_load(open("configs/tuned_heuristic.yaml"))
models = {f"learned-s{i}": load(p) for i, p in enumerate(["runs/ours_dfm_final.pt", "runs/ours_dfm_final_s1.pt", "runs/ours_dfm_final_s2.pt", "runs/ours_dfm_final_s3.pt"])}
topo, trace = real.load(net)
out, t0 = {"network": net, "windows": W, "cells": {}}, time.time()
seeds = [20000 + i for i in range(W)]
for ld in (0.6, 0.8, 1.0):
    for pn in ("ideal", "rich", "snmp", "degraded"):
        cfg = EnvCfg(load=ld, horizon=100, telemetry=PROFILES[pn], trace=trace)
        cell = {}
        pols = {"tuned": TunedHeuristic(**tuned[pn]["params"]), **{k: LearnedPolicy(m, c, k) for k, (m, c) in models.items()}}
        for k, pol in pols.items():
            rows = [episode_stats(topo, cfg, pol, s) for s in seeds]
            cell[k] = [float(1 - r["mu"] / r["ecmp_mu"]) if r["ecmp_mu"] > 1e-9 else float("nan") for r in rows]   # NaN = zero-traffic window
        out["cells"][f"{ld}|{pn}"] = cell
        print(net, ld, pn, f"t={time.time() - t0:.0f}s", flush=True)
        json.dump(out, open(f"experiments/final/realci_{net}.json", "w"))
