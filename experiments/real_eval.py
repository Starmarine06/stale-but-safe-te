"""Main real-trace evaluation (SNDlib traces): per network x load x telemetry profile.

  python experiments/real_eval.py abilene --out experiments/final/real_abilene.json [--seeds 4]

Methods: ecmp, tuned, gnn-ours-s{0..3} (randomised-telemetry training seeds), gnn-ideal-s0,
gnn-ours-s0+guard (L2-only, eps=-0.06, chosen on validation) and tuned+guard, oracle (n<=25).
Rows use the same fields as final_eval.py with topo = "<network>@<load>" so analyze_final.py works.
Each eval seed replays a different random window of the real trace.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from stale_but_safe.baselines.heuristics import ECMP, LPOracle, TunedHeuristic
from stale_but_safe.env import real
from stale_but_safe.env.te_env import EnvCfg
from stale_but_safe.env.telemetry import PROFILES
from stale_but_safe.eval.compare import ecmp_score, episode_stats, score
from stale_but_safe.guard.guard import Guard, GuardCfg
from stale_but_safe.models.learned import LearnedPolicy, load

ap = argparse.ArgumentParser()
ap.add_argument("network")
ap.add_argument("--out", required=True)
ap.add_argument("--seeds", type=int, default=4)
ap.add_argument("--horizon", type=int, default=100)
ap.add_argument("--loads", default="0.6,0.8,1.0")
ap.add_argument("--profiles", default="ideal,rich,snmp,degraded")
a = ap.parse_args()

tuned = yaml.safe_load(open("configs/tuned_heuristic.yaml"))
G2 = GuardCfg(eps=-0.06, layers=(True, True, False))
paths = {"gnn-ours-s0": "runs/ours_dfm_final.pt", "gnn-ideal-s0": "runs/ideal_dfm_final.pt"}
for s in (1, 2, 3):
    if Path(f"runs/ours_dfm_final_s{s}.pt").exists():
        paths[f"gnn-ours-s{s}"] = f"runs/ours_dfm_final_s{s}.pt"
models = {k: load(p) for k, p in paths.items()}
pol_gnn = lambda k: LearnedPolicy(*models[k], k)

topo, trace = real.load(a.network)
out = {"args": vars(a), "results": []}
seeds = [5000 + i for i in range(a.seeds)]
t0 = time.time()
for ld in [float(x) for x in a.loads.split(",")]:
    oracle_rec = None
    for pn in a.profiles.split(","):
        cfg = EnvCfg(load=ld, horizon=a.horizon, telemetry=PROFILES[pn], trace=trace)
        meth = {"ecmp": ECMP(), "tuned": TunedHeuristic(**tuned[pn]["params"]),
                "tuned+guard": Guard(TunedHeuristic(**tuned[pn]["params"]), G2)}
        for k in models:
            meth[k] = pol_gnn(k)
        meth["gnn-ours-s0+guard"] = Guard(pol_gnn("gnn-ours-s0"), G2)
        todo = dict(meth)
        if topo.n <= 25:
            todo["oracle"] = LPOracle()
        for mname, pol in todo.items():
            if mname == "oracle" and oracle_rec is not None:       # oracle sees true demand: profile-independent
                out["results"].append({**oracle_rec, "profile": pn})
                continue
            rows = [episode_stats(topo, cfg, pol, s) for s in seeds]
            reg = np.concatenate([r["regret"] for r in rows])
            rec = {"topo": f"{a.network}@{ld}", "n": topo.n, "profile": pn, "method": mname,
                   "delta": float(np.mean([score(r) - ecmp_score(r) for r in rows])),
                   "gain": float(np.mean([1 - r["mu"] / r["ecmp_mu"] for r in rows])),
                   "loss": float(np.mean([r["loss"] for r in rows])),
                   "churn": float(np.mean([r["churn"] for r in rows])),
                   "regret_p99": float(np.quantile(reg, 0.99)), "regret_mean": float(reg.mean()),
                   "regret_cvar95": float(np.sort(reg)[-max(1, len(reg) // 20):].mean())}
            if hasattr(pol, "stats"):
                rec["fallback_rate"] = pol.stats["fallback"] / max(pol.stats["steps"], 1)
            if mname == "oracle":
                oracle_rec = rec
            out["results"].append(rec)
        print(f"{a.network} load={ld} {pn:8s} done t={time.time() - t0:.0f}s", flush=True)
        json.dump(out, open(a.out, "w"))
json.dump(out, open(a.out, "w"))
