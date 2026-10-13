"""What does the guard actually do? Per interval, compare the policy's raw proposal with ECMP using the TRUE demand
(counterfactual, simulator only) and cross-tabulate with the guard's decision.

  python experiments/guard_audit.py            -> experiments/final/table_guard_audit.md

Definitions (per interval, 'beneficial' = proposal's true max-util < 0.99 x ECMP's, 'harmful' = > 1.01 x ECMP's):
  fallback rate        share of intervals the guard replaced the proposal with ECMP
  false rejection      fallbacks where the proposal was beneficial  / all fallbacks
  harmful passed       accepted intervals where the proposal was harmful / all accepted
  harm caught          harmful proposals that the guard replaced / all harmful proposals
  gain kept            (ECMP mu - guarded mu) / (ECMP mu - raw mu)
"""
import sys
import numpy as np

from stale_but_safe.baselines.heuristics import Policy
from stale_but_safe.env import real
from stale_but_safe.env.te_env import EnvCfg, TEEnv
from stale_but_safe.env.telemetry import PROFILES
from stale_but_safe.env.traffic import ecmp_splits
from stale_but_safe.env.zoo import load_split, zoo_topology
from stale_but_safe.guard.guard import Guard, GuardCfg
from stale_but_safe.models.learned import LearnedPolicy, load

m, c = load("runs/ours_dfm_final.pt")
CONFIGS = {"L2-only eps=-0.06 (headline)": GuardCfg(eps=-0.06, layers=(True, True, False)),
           "L2-only eps=0": GuardCfg(eps=0.0, layers=(True, True, False)),
           "all layers eps=0, tau=0.05 (default)": GuardCfg()}


class Spy(Policy):
    name = "spy"

    def __init__(self, inner):
        self.inner, self.last = inner, None

    def reset(self, env):
        super().reset(env); self.inner.reset(env)

    def act(self, obs, env):
        self.last = np.asarray(self.inner.act(obs, env), float)
        return self.last


def mu(topo, dem, splits, up):
    u = topo.inc @ (dem[:, None] * splits).ravel() / topo.cap
    return float(u[up].max())


def run(topo, cfg, gcfg, seed):
    env = TEEnv(topo, cfg, seed)
    obs = env.reset(seed)
    spy = Spy(LearnedPolicy(m, c, "gnn"))
    g = Guard(spy, gcfg)
    g.reset(env)
    rec, done = [], False
    while not done:
        dem, up = env.peek_demand(), env.link_up.copy()
        act = g.act(obs, env)
        ecmp = ecmp_splits(topo, topo.valid_paths(up))
        prop = spy.last if spy.last is not None else act
        fb = spy.last is None or not np.allclose(act, spy.last, atol=1e-9)   # spy not called => L1/L3 fallback
        rec.append((mu(topo, dem, prop, up), mu(topo, dem, act, up), mu(topo, dem, ecmp, up), fb))
        spy.last = None
        obs, _, done, _ = env.step(act)
    return np.array(rec, float)


def summarise(R):
    mp, ma, me, fb = R[:, 0], R[:, 1], R[:, 2], R[:, 3].astype(bool)
    ben, harm = mp < 0.99 * me, mp > 1.01 * me
    fr = lambda a, b: float(a.sum() / b.sum()) if b.sum() else float("nan")
    return {"fallback": fb.mean(), "false_rej": fr(fb & ben, fb), "harmful_passed": fr(~fb & harm, ~fb),
            "harm_caught": fr(fb & harm, harm), "gain_kept": float((me - ma).sum() / max((me - mp).sum(), 1e-9)),
            "p99_raw": float(np.quantile(mp - me, .99)), "p99_guard": float(np.quantile(ma - me, .99)),
            "n": len(R), "harm_share": harm.mean(), "ben_share": ben.mean()}


cases = []
zoo = [zoo_topology(n) for n in load_split("test_subset")]
zoo = [t for t in zoo if t.n <= 25][:10]
for prof in ("ideal", "degraded"):
    cases.append((f"Zoo (10 graphs, n<=25) {prof}", [(t, EnvCfg(load=0.8, horizon=100, telemetry=PROFILES[prof])) for t in zoo]))
for scen, kw in (("link failure", dict(fail_at=40)), ("demand shift", dict(traffic={"shift_at": 50}))):
    cases.append((f"Zoo {scen} degraded", [(t, EnvCfg(load=0.8, horizon=100, telemetry=PROFILES["degraded"], **kw)) for t in zoo]))
for net in ("abilene", "geant"):
    topo, trace = real.load(net)
    for prof in ("ideal", "degraded"):
        cases.append((f"{net} real trace {prof}", [(topo, EnvCfg(load=0.8, horizon=100, telemetry=PROFILES[prof], trace=trace))]))

print("| case | guard | fallback | false rejection | harmful passed | harm caught | gain kept | p99 regret raw -> guarded |")
print("|---|---|---|---|---|---|---|---|")
for name, items in cases:
    for gn, gcfg in CONFIGS.items():
        R = np.concatenate([run(t, cfg, gcfg, s) for t, cfg in items for s in range(3)])
        s = summarise(R)
        print(f"| {name} | {gn} | {100 * s['fallback']:.0f}% | {100 * s['false_rej']:.0f}% | {100 * s['harmful_passed']:.1f}% "
              f"| {100 * s['harm_caught']:.0f}% | {100 * s['gain_kept']:.0f}% | {s['p99_raw']:+.3f} -> {s['p99_guard']:+.3f} |", flush=True)
