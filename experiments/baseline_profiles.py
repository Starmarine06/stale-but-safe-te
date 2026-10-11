"""Heuristic baselines across telemetry profiles (preview of RQ1 for non-learning methods)."""
from stale_but_safe.env.te_env import EnvCfg
from stale_but_safe.env.telemetry import PROFILES
from stale_but_safe.env.topology import synthetic
from stale_but_safe.baselines.heuristics import ECMP, Greedy, TunedHeuristic, LPOracle
from stale_but_safe.eval.runner import evaluate

topo = synthetic("ba", 12, 1)
seeds = range(6)
print(f"{'profile':9s} {'policy':8s} {'gain%':>6s} {'p99regret':>9s} {'loss':>6s} {'churn':>6s}")
for pname, tele in PROFILES.items():
    cfg = EnvCfg(load=0.9, horizon=100, telemetry=tele)
    for pol in (ECMP(), Greedy(), TunedHeuristic(), LPOracle()):
        r = evaluate(topo, cfg, pol, seeds)
        print(f"{pname:9s} {pol.name:8s} {100*r['improvement_vs_ecmp']:6.1f} {r['regret_p99']:9.3f} {r['mean_loss']:6.3f} {r['churn']:6.3f}")
