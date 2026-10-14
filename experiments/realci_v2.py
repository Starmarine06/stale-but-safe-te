"""Recompute the *tuned* column of realci_<net>.json with the re-tuned per-profile heuristic (configs/tuned_heuristic_v2.yaml), same windows (seeds 20000+),
same cells; learned columns are copied unchanged. -> experiments/final/realci_v2_<net>.json
  python experiments/realci_v2.py abilene [W]"""
import json, sys
import yaml
from stale_but_safe.baselines.heuristics import TunedHeuristic
from stale_but_safe.env import real
from stale_but_safe.env.te_env import EnvCfg
from stale_but_safe.env.telemetry import PROFILES
from stale_but_safe.eval.compare import episode_stats

net = sys.argv[1]
old = json.load(open(f"experiments/final/realci_{net}.json"))
W = old["windows"]
v2 = yaml.safe_load(open("configs/tuned_heuristic_v2.yaml"))
topo, trace = real.load(net)
seeds = [20000 + i for i in range(W)]
out = {"network": net, "windows": W, "cells": {}}
for key, cell in old["cells"].items():
    ld, pn = key.split("|")
    cfg = EnvCfg(load=float(ld), horizon=100, telemetry=PROFILES[pn], trace=trace)
    pol = TunedHeuristic(**v2[pn]["params"])
    rows = [episode_stats(topo, cfg, pol, s) for s in seeds]
    new = dict(cell)
    new["tuned_old"] = cell["tuned"]
    new["tuned"] = [float(1 - r["mu"] / r["ecmp_mu"]) if r["ecmp_mu"] > 1e-9 else float("nan") for r in rows]
    out["cells"][key] = new
    print(net, key, flush=True)
    json.dump(out, open(f"experiments/final/realci_v2_{net}.json", "w"))
