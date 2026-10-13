"""Sanity check on REAL SNDlib traces: ECMP vs tuned vs LP oracle vs learned, per telemetry profile."""
import sys
import numpy as np
import yaml

from stale_but_safe.baselines.heuristics import ECMP, LPOracle, TunedHeuristic
from stale_but_safe.env import real
from stale_but_safe.env.te_env import EnvCfg
from stale_but_safe.env.telemetry import PROFILES
from stale_but_safe.eval.runner import evaluate
from stale_but_safe.models.learned import LearnedPolicy, load

names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["abilene", "geant", "germany17"]
load_lvl = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
tuned = yaml.safe_load(open("configs/tuned_heuristic.yaml"))
m, c = load("runs/ours_dfm_final.pt")
for n in names:
    topo, trace = real.load(n)
    print(f"== {n}: nodes={topo.n} links={topo.E} demands={topo.D} T={len(trace)}", flush=True)
    for prof in ("ideal", "snmp", "degraded"):
        cfg = EnvCfg(load=load_lvl, horizon=100, telemetry=PROFILES[prof], trace=trace)
        pols = {"ecmp": ECMP(), "tuned": TunedHeuristic(**tuned[prof]["params"]), "gnn": LearnedPolicy(m, c, "gnn")}
        if prof == "ideal" and topo.n <= 25:
            pols["oracle"] = LPOracle()
        res = {k: evaluate(topo, cfg, p, range(3)) for k, p in pols.items()}
        print(prof, " ".join(f"{k}:{100 * r['improvement_vs_ecmp']:.1f}% p99={r['regret_p99']:+.3f}" for k, r in res.items()), flush=True)
