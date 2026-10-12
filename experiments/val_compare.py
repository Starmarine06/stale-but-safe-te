"""Validation comparison: heuristics / oracle / learned (ideal vs randomized) / guard,
across telemetry profiles on the 12 validation Zoo graphs + 3 held-out synthetic graphs.
Usage: python experiments/val_compare.py runs/ideal_s0.pt runs/ours_s0.pt [eps]"""
import sys

import yaml

from stale_but_safe.baselines.heuristics import ECMP, Greedy, LPOracle, TunedHeuristic
from stale_but_safe.env.telemetry import PROFILES
from stale_but_safe.env.topology import synthetic
from stale_but_safe.env.zoo import load_split, zoo_topology
from stale_but_safe.eval.compare import print_table, summarize, table
from stale_but_safe.guard.guard import Guard, GuardCfg
from stale_but_safe.models.learned import LearnedPolicy, load

ideal_path, ours_path = sys.argv[1], sys.argv[2]
eps = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
tuned = yaml.safe_load(open("configs/tuned_heuristic.yaml"))
topos = [synthetic("ba", 14, 8001), synthetic("er", 16, 8002), synthetic("geo", 15, 8003)]
topos += [zoo_topology(n) for n in load_split("val")]
mi, ci = load(ideal_path)
mo, co = load(ours_path)
policies = {
    "ecmp": lambda p, t: ECMP(),
    "greedy": lambda p, t: Greedy(),
    "tuned": lambda p, t: TunedHeuristic(**tuned[p]["params"]),
    "gnn-ideal": lambda p, t: LearnedPolicy(mi, ci, "gnn-ideal"),
    "gnn-rand": lambda p, t: LearnedPolicy(mo, co, "gnn-rand"),
    "rand+guard": lambda p, t: Guard(LearnedPolicy(mo, co, "gnn-rand"), GuardCfg(eps=eps)),
    "ideal+guard": lambda p, t: Guard(LearnedPolicy(mi, ci, "gnn-ideal"), GuardCfg(eps=eps)),
    "oracle": lambda p, t: LPOracle(),
}
res = table(topos, policies, PROFILES, seeds=[5000], horizon=60)
print_table(summarize(res))
