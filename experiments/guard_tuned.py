"""Guard around the TUNED HEURISTIC (no torch needed): shows the guard bounds regret of a policy
that is worse than ECMP under degraded telemetry, and maps the gain-vs-p99-regret trade-off."""
import sys

import numpy as np
import yaml

from stale_but_safe.baselines.heuristics import TunedHeuristic
from stale_but_safe.env.telemetry import PROFILES
from stale_but_safe.env.topology import synthetic
from stale_but_safe.env.zoo import load_split, zoo_topology
from stale_but_safe.eval.compare import table
from stale_but_safe.eval.stats import cvar
from stale_but_safe.guard.guard import Guard, GuardCfg

tuned = yaml.safe_load(open("configs/tuned_heuristic.yaml"))
topos = [synthetic("ba", 14, 8001), synthetic("er", 16, 8002), synthetic("geo", 15, 8003)]
topos += [zoo_topology(n) for n in load_split("val")]
profs = {p: PROFILES[p] for p in ("snmp", "degraded")}
mk = lambda p: TunedHeuristic(**tuned[p]["params"])
V = {"raw": lambda p, t: mk(p)}
for eps in (0.05, 0.0, -0.03):
    V[f"L123 eps={eps}"] = (lambda e: lambda p, t: Guard(mk(p), GuardCfg(eps=e)))(eps)
for tau in (0.02, 0.05, 0.1):
    V[f"L3 tau={tau}"] = (lambda g: lambda p, t: Guard(mk(p), GuardCfg(layers=(False, False, True), tau=g)))(tau)
V["L1 only"] = lambda p, t: Guard(mk(p), GuardCfg(layers=(True, False, False)))
V["L2 only eps=0"] = lambda p, t: Guard(mk(p), GuardCfg(layers=(False, True, False), eps=0.0))
res = table(topos, V, profs, seeds=[5000, 5001], horizon=60)
print(f"{'profile':9s} {'variant':16s} {'dScore':>8s} {'gain%':>7s} {'p99reg':>8s} {'cvar95':>8s}")
for (pn, name), rows in res.items():
    reg = np.concatenate([r["regret"] for r in rows])
    print(f"{pn:9s} {name:16s} {np.mean([r['delta'] for r in rows]):8.3f} "
          f"{100*np.mean([r['gain'] for r in rows]):7.1f} {np.quantile(reg, .99):8.3f} {cvar(reg):8.3f}")
