import numpy as np
import pytest

from stale_but_safe.baselines.heuristics import (ECMP, SPF, Greedy, LPOracle,
                                                 TunedHeuristic, solve_min_maxutil)
from stale_but_safe.env.te_env import EnvCfg, TEEnv
from stale_but_safe.env.telemetry import Channel, TelemetryCfg
from stale_but_safe.env.topology import abilene, synthetic
from stale_but_safe.env.traffic import ecmp_splits
from stale_but_safe.eval.runner import evaluate


@pytest.fixture(scope="module")
def topo():
    return abilene(k=4)


# ---------------------------------------------------------------- topology
def test_abilene_shape(topo):
    assert topo.n == 12 and topo.E == 30 and topo.D == 132
    assert topo.path_mask[:, 0].all()          # every demand has a path
    assert (topo.inc.sum(0).reshape(topo.D, topo.k)[topo.path_mask] ==
            topo.hops[topo.path_mask]).all()   # column sums = hop counts


def test_paths_are_loop_free_and_connect(topo):
    for d, (s, t) in enumerate(topo.demands):
        for ls in topo.path_links[d]:
            nodes = [topo.links[ls[0]][0]] + [topo.links[l][1] for l in ls]
            assert nodes[0] == s and nodes[-1] == t
            assert len(set(nodes)) == len(nodes)


def test_synthetic_connected():
    t = synthetic("ba", 15, seed=3)
    assert t.path_mask[:, 0].all()


def test_ecmp_rows_sum_to_one(topo):
    s = ecmp_splits(topo)
    assert np.allclose(s.sum(1), 1.0)


# --------------------------------------------------------------- telemetry
def test_ideal_channel_is_identity():
    ch = Channel(3, TelemetryCfg(), np.random.default_rng(0))
    for i in range(5):
        x = np.full(3, float(i))
        ch.push(x)
        v, a = ch.observe()
        assert np.allclose(v, x) and np.allclose(a, 0)


def test_delay_is_exact():
    ch = Channel(2, TelemetryCfg(P=1, d=3), np.random.default_rng(0))
    for i in range(10):
        ch.push(np.full(2, float(i)))
    v, a = ch.observe()
    assert np.allclose(v, 6.0) and np.allclose(a, 3)   # sample from 3 steps ago


def test_polling_period_staleness():
    ch = Channel(1, TelemetryCfg(P=5, d=0), np.random.default_rng(0))
    seen, ages = [], []
    for i in range(12):
        ch.push(np.array([float(i)]))
        v, a = ch.observe()
        seen.append(v[0]); ages.append(a[0])
    assert seen[:5] == [0, 0, 0, 0, 0] and seen[5] == 5
    assert ages[:5] == [0, 1, 2, 3, 4]


def test_noise_is_mean_preserving_and_drops_match_rate():
    rng = np.random.default_rng(1)
    ch = Channel(20000, TelemetryCfg(P=1, d=0, sigma=0.3, p=0.25), rng)
    ch.push(np.ones(20000))                    # first poll never drops
    ch.push(np.ones(20000))
    v, a = ch.observe()
    assert abs((a > 0).mean() - 0.25) < 0.02   # dropped entries are stale
    assert abs(v.mean() - 1.0) < 0.02


# --------------------------------------------------------------------- env
def test_env_deterministic_given_seed(topo):
    cfg = EnvCfg(horizon=30)
    a = evaluate(topo, cfg, ECMP(), [3])["raw"]["max_util"]
    b = evaluate(topo, cfg, ECMP(), [3])["raw"]["max_util"]
    assert np.array_equal(a, b)


def test_ecmp_has_zero_regret(topo):
    r = evaluate(topo, EnvCfg(horizon=40), ECMP(), [0, 1])
    assert r["regret_mean"] == pytest.approx(0, abs=1e-9)


def test_sanitize_always_valid(topo):
    env = TEEnv(topo, EnvCfg(), 0)
    bad = np.random.default_rng(0).normal(size=(topo.D, topo.k))
    bad[0] = np.nan
    bad[1] = 0
    s = env.sanitize(bad)
    assert np.allclose(s.sum(1), 1) and (s >= 0).all()
    assert (s[~topo.path_mask] == 0).all()


def test_link_failure_reroutes(topo):
    env = TEEnv(topo, EnvCfg(fail_at=0, horizon=5), 0)
    env.step(ecmp_splits(topo))
    valid = topo.valid_paths(env.link_up)
    s = env.sanitize(ecmp_splits(topo))
    assert (s[~valid] == 0).all()
    assert not env.link_up.all()


def test_install_latency_delays_action(topo):
    cfg = EnvCfg(L=2, horizon=10)
    env = TEEnv(topo, cfg, 0)
    spf = SPF(); spf.reset(env)
    s = spf.act(env.observe(), env)
    env.step(s); assert np.allclose(env.applied, ecmp_splits(topo))   # not yet
    env.step(s); assert np.allclose(env.applied, ecmp_splits(topo))
    env.step(s); assert np.allclose(env.applied, s)                   # after L=2


# --------------------------------------------------------------- baselines
@pytest.mark.parametrize("pol", [ECMP, SPF, Greedy, TunedHeuristic, LPOracle])
def test_policies_emit_valid_splits(topo, pol):
    env = TEEnv(topo, EnvCfg(fail_at=3), 1)
    p = pol(); obs = env.reset(1); p.reset(env)
    for _ in range(8):
        s = p.act(obs, env)
        assert s.shape == (topo.D, topo.k)
        assert (s >= -1e-9).all()
        obs, *_ = env.step(s)


def test_oracle_beats_ecmp_on_max_util(topo):
    env = TEEnv(topo, EnvCfg(load=0.8), 0)
    dem = env.peek_demand()
    valid = topo.valid_paths()
    s, z = solve_min_maxutil(topo, dem, valid)
    e = ((topo.inc @ (dem[:, None] * ecmp_splits(topo)).ravel()) / topo.cap).max()
    assert z <= e + 1e-6
    lp_u = ((topo.inc @ (dem[:, None] * s).ravel()) / topo.cap).max()
    assert lp_u == pytest.approx(z, rel=1e-4)


def test_oracle_headroom_gate1():
    """Gate 1 smoke check. Headroom exists on heterogeneous synthetic graphs
    but is small on Abilene (see experiments/gate1_headroom)."""
    cfg = EnvCfg(load=0.9, horizon=20)
    syn = evaluate(synthetic("ba", 12, 1), cfg, LPOracle(), [0])
    assert syn["improvement_vs_ecmp"] > 0.10
    ab = evaluate(abilene(), cfg, LPOracle(), [0])
    assert 0.0 <= ab["improvement_vs_ecmp"] < 0.10
