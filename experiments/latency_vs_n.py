"""Decision latency per interval vs network size (single CPU thread, p50/p99 over 100 calls after warm-up).
Synthetic BA graphs with all-pairs demands (D = n(n-1)) and Brain (161 nodes, 600 demands)."""
import time
import numpy as np
import torch
import yaml

from stale_but_safe.baselines.heuristics import TunedHeuristic
from stale_but_safe.env import real
from stale_but_safe.env.te_env import EnvCfg, TEEnv
from stale_but_safe.env.topology import synthetic
from stale_but_safe.guard.guard import Guard, GuardCfg
from stale_but_safe.models.learned import LearnedPolicy, load

torch.set_num_threads(1)
tuned = yaml.safe_load(open("configs/tuned_heuristic.yaml"))
m, c = load("runs/ours_dfm_final.pt")
print("| network | nodes | links | demands | tuned ms (p50/p99) | learned ms (p50/p99) | learned+guard ms (p50/p99) | peak extra memory MB |")
print("|---|---|---|---|---|---|---|---|")
import tracemalloc


def timed(pol, topo, cfg):
    env = TEEnv(topo, cfg, 0); obs = env.reset(0); pol.reset(env)
    ts = []
    for i in range(110):
        t0 = time.perf_counter(); a = pol.act(obs, env); ts.append((time.perf_counter() - t0) * 1000)
        obs, _, done, _ = env.step(a)
        if done:
            obs = env.reset(0); pol.reset(env)
    ts = np.array(ts[10:]); return np.percentile(ts, 50), np.percentile(ts, 99)


cases = [(f"BA{n}", synthetic("ba", n, 1)) for n in (12, 25, 50)] + [("BA80", synthetic("ba", 80, 1))]
cases.append(("Brain (real, 600 demands)", real.load("brain")[0]))
for name, topo in cases:
    cfg = EnvCfg(load=0.8, horizon=130)
    tracemalloc.start()
    a = timed(TunedHeuristic(**tuned["snmp"]["params"]), topo, cfg)
    b = timed(LearnedPolicy(m, c, "g"), topo, cfg)
    g = timed(Guard(LearnedPolicy(m, c, "g"), GuardCfg(eps=-0.06, layers=(True, True, False))), topo, cfg)
    peak = tracemalloc.get_traced_memory()[1] / 1e6; tracemalloc.stop()
    print(f"| {name} | {topo.n} | {topo.E} | {topo.D} | {a[0]:.1f} / {a[1]:.1f} | {b[0]:.1f} / {b[1]:.1f} | {g[0]:.1f} / {g[1]:.1f} | {peak:.0f} |", flush=True)
