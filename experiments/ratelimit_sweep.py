"""Operational cost trade-off: gain over ECMP vs flow updates per interval as the rate-limit threshold grows."""
import numpy as np
import yaml

from stale_but_safe.baselines.heuristics import TunedHeuristic
from stale_but_safe.baselines.ratelimit import RateLimited
from stale_but_safe.env import real
from stale_but_safe.env.te_env import EnvCfg
from stale_but_safe.env.telemetry import PROFILES
from stale_but_safe.env.zoo import load_split, zoo_topology
from stale_but_safe.eval.compare import episode_stats
from stale_but_safe.models.learned import LearnedPolicy, load

tuned = yaml.safe_load(open("configs/tuned_heuristic.yaml"))
m, c = load("runs/ours_dfm_final.pt")
zoo = [t for t in (zoo_topology(n) for n in load_split("test_subset")) if t.n <= 25][:10]
abil, atr = real.load("abilene"); gea, gtr = real.load("geant")
cases = [("Zoo (10 graphs)", zoo, None), ("Abilene real", [abil], atr), ("GEANT real", [gea], gtr)]
import sys
if len(sys.argv) > 1:   # e.g. "real" -> only the real-trace cases
    cases = [c for c in cases if (("real" in c[0]) == (sys.argv[1] == "real"))]
H = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
print("| case | telemetry | method | threshold | gain % | updates / demand / interval | p99 regret |")
print("|---|---|---|---|---|---|---|")
for name, topos, trace in cases:
    for prof in ("ideal", "degraded"):
        for mname, mk in (("learned", lambda: LearnedPolicy(m, c, "gnn")),
                          ("tuned", lambda: TunedHeuristic(**tuned[prof]["params"]))):
            for h in H:
                gains, upd, reg = [], [], []
                for t in topos:
                    cfg = EnvCfg(load=0.8, horizon=100, telemetry=PROFILES[prof], trace=trace)
                    for s in range(4):
                        pol = RateLimited(mk(), h)
                        r = episode_stats(t, cfg, pol, 300 + s)
                        gains.append(1 - r["mu"] / r["ecmp_mu"]); reg.append(r["regret"])
                        upd.append(np.mean(pol.updates) / t.D)
                print(f"| {name} | {prof} | {mname} | {h} | {100 * np.mean(gains):.1f} | {np.mean(upd):.3f} "
                      f"| {np.quantile(np.concatenate(reg), .99):+.3f} |", flush=True)
