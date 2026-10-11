import numpy as np
import pytest

from stale_but_safe.baselines.heuristics import ECMP, Policy
from stale_but_safe.env.te_env import EnvCfg, TEEnv
from stale_but_safe.env.telemetry import TelemetryCfg
from stale_but_safe.env.topology import synthetic
from stale_but_safe.env.traffic import ecmp_splits
from stale_but_safe.eval.runner import evaluate
from stale_but_safe.guard.guard import Guard, GuardCfg


@pytest.fixture(scope="module")
def topo():
    return synthetic("ba", 12, 1)


class Bad(Policy):
    """Always dumps every demand on its first path (terrible at load) or emits junk."""
    name = "bad"

    def __init__(self, mode="spf"):
        self.mode = mode

    def act(self, obs, env):
        d, k = self.topo.D, self.topo.k
        if self.mode == "nan":
            return np.full((d, k), np.nan)
        if self.mode == "neg":
            return -np.ones((d, k))
        out = np.zeros((d, k)); out[:, 0] = 1
        return out


def run(topo, pol, cfg=None, steps=30):
    env = TEEnv(topo, cfg or EnvCfg(load=0.9, horizon=steps), 0)
    obs = env.reset(0); pol.reset(env)
    acts = []
    for _ in range(steps):
        a = pol.act(obs, env); acts.append(a)
        obs, *_ = env.step(a)
    return acts, pol


@pytest.mark.parametrize("mode", ["nan", "neg"])
def test_l1_blocks_malformed(topo, mode):
    g = Guard(Bad(mode), GuardCfg(layers=(True, False, False)))
    acts, g = run(topo, g, steps=5)
    for a in acts:
        assert np.isfinite(a).all() and (a >= 0).all() and np.allclose(a.sum(1), 1)
    assert g.stats["L1"] == 5


def test_l1_stale_telemetry_falls_back(topo):
    cfg = EnvCfg(load=0.9, horizon=40, telemetry=TelemetryCfg(P=30, d=0))
    g = Guard(Bad("spf"), GuardCfg(layers=(True, False, False), max_mean_age=5))
    acts, g = run(topo, g, cfg, 40)
    assert g.stats["L1"] > 10


def test_guard_bounds_regret_of_bad_policy(topo):
    cfg = EnvCfg(load=0.9, horizon=60)
    raw = evaluate(topo, cfg, Bad("spf"), [0, 1])
    grd = evaluate(topo, cfg, Guard(Bad("spf"), GuardCfg(eps=0.0)), [0, 1])
    assert grd["regret_p99"] <= max(raw["regret_p99"], 0)
    assert grd["regret_mean"] <= raw["regret_mean"] + 1e-9
    assert grd["regret_p99"] < 0.2          # stays near ECMP


def test_l3_latches_when_estimates_are_wrong(topo):
    """L2 disabled: monitor alone must detect a persistently bad policy and latch."""
    cfg = EnvCfg(load=0.9, horizon=80)
    g = Guard(Bad("spf"), GuardCfg(layers=(False, False, True), tau=0.02, k_consec=2, window=3))
    _, g = run(topo, g, cfg, 80)
    assert g.stats["L3"] > 0


def test_guard_with_ecmp_never_falls_back(topo):
    g = Guard(ECMP(), GuardCfg())
    _, g = run(topo, g, steps=30)
    assert g.stats["L2"] == 0 and g.stats["L1"] == 0


def test_guard_actions_always_valid_under_failure(topo):
    cfg = EnvCfg(load=0.9, horizon=40, fail_at=10, telemetry=TelemetryCfg(P=3, d=2, sigma=0.2, p=0.1))
    g = Guard(Bad("spf"))
    acts, _ = run(topo, g, cfg, 40)
    env = TEEnv(topo, cfg, 0)
    for a in acts:
        assert np.isfinite(a).all() and (a >= -1e-9).all()
