"""Guard epsilon sweep + layer ablation: gain vs worst-case regret (RQ3).
Usage: python experiments/guard_sweep.py <model.pt> [profile ...]"""
import sys

import numpy as np

from stale_but_safe.env.telemetry import PROFILES
from stale_but_safe.env.topology import synthetic
from stale_but_safe.env.zoo import load_split, zoo_topology
from stale_but_safe.eval.compare import table
from stale_but_safe.eval.stats import cvar
from stale_but_safe.guard.guard import Guard, GuardCfg
from stale_but_safe.models.learned import LearnedPolicy, load

model, cfg = load(sys.argv[1])
profs = {p: PROFILES[p] for p in (sys.argv[2:] or ["snmp", "degraded"])}
topos = [synthetic("ba", 14, 8001), synthetic("er", 16, 8002), synthetic("geo", 15, 8003)]
topos += [zoo_topology(n) for n in load_split("val")]
mk = lambda: LearnedPolicy(model, cfg, "gnn")
variants = {"raw": lambda p, t: mk()}
for eps in (0.0, 0.05, 0.1, 0.2, 0.5):
    variants[f"L123 eps={eps}"] = (lambda e: lambda p, t: Guard(mk(), GuardCfg(eps=e)))(eps)
for name, layers in (("L1 only", (1, 0, 0)), ("L2 only (eps=.1)", (0, 1, 0)), ("L3 only", (0, 0, 1)), ("L1+L3", (1, 0, 1))):
    variants[name] = (lambda l: lambda p, t: Guard(mk(), GuardCfg(eps=0.1, layers=tuple(map(bool, l)))))(layers)
res = table(topos, variants, profs, seeds=[5000, 5001], horizon=60)
print(f"{'profile':9s} {'variant':18s} {'dScore':>8s} {'gain%':>7s} {'p99reg':>8s} {'cvar95':>8s}")
for (pn, name), rows in res.items():
    reg = np.concatenate([r["regret"] for r in rows])
    print(f"{pn:9s} {name:18s} {np.mean([r['delta'] for r in rows]):8.3f} "
          f"{100*np.mean([r['gain'] for r in rows]):7.1f} {np.quantile(reg, .99):8.3f} {cvar(reg):8.3f}")
