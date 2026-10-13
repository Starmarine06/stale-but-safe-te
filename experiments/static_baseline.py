"""Static routing baseline: LP-optimal splits for the MEAN traffic matrix of the training part of the trace (no telemetry
needed, never changes unless a link fails). Evaluated on the same held-out windows/seeds as the DOTE-style baseline.
If this matches DOTE-style and 'learned' gains, robustness to telemetry comes from traffic stationarity, not learning."""
import json
import sys

import numpy as np

from stale_but_safe.baselines.heuristics import ECMP, Policy, solve_min_maxutil
from stale_but_safe.env import real
from stale_but_safe.env.te_env import EnvCfg
from stale_but_safe.env.telemetry import PROFILES
from stale_but_safe.eval.compare import ecmp_score, episode_stats, score


class StaticLP(Policy):
    name = "static-lp"

    def __init__(self, ref):
        self.ref, self.cache = ref, {}

    def act(self, obs, env):
        valid = self.topo.valid_paths(obs["link_up"])
        key = valid.tobytes()
        if key not in self.cache:
            self.cache[key] = solve_min_maxutil(self.topo, self.ref, valid)[0]
        return self.cache[key]


net = sys.argv[1]
topo, trace = real.load(net)
cut = int(0.7 * len(trace))
ref = trace[:cut].mean(0)
te = trace[cut:]
out = {"results": []}
seeds = [9000 + i for i in range(10)]
for ld in (0.6, 0.8, 1.0):
    cfg = EnvCfg(load=ld, horizon=100, telemetry=PROFILES["ideal"], trace=te)
    pol = StaticLP(ref)
    rows = [episode_stats(topo, cfg, pol, s) for s in seeds]
    reg = np.concatenate([r["regret"] for r in rows])
    for pn in ("ideal", "snmp", "degraded"):
        out["results"].append({"topo": f"{net}-test@{ld}", "n": topo.n, "profile": pn, "method": "static-lp",
                               "gain": float(np.mean([1 - r["mu"] / r["ecmp_mu"] for r in rows])),
                               "delta": float(np.mean([score(r) - ecmp_score(r) for r in rows])),
                               "regret_p99": float(np.quantile(reg, 0.99))})
    print(net, ld, f"static-lp gain {100 * out['results'][-1]['gain']:.1f}% p99 {out['results'][-1]['regret_p99']:+.3f}", flush=True)
json.dump(out, open(f"experiments/final/static_{net}.json", "w"))
