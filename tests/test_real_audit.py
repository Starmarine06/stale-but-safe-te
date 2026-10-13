import numpy as np
import pytest

from stale_but_safe import audit as A
from stale_but_safe.env import real
from stale_but_safe.env.te_env import EnvCfg, TEEnv
from stale_but_safe.env.telemetry import TelemetryCfg
from stale_but_safe.env.topology import synthetic
from stale_but_safe.env.traffic import ecmp_max_util

have_real = (real.ROOT / "abilene.npz").exists()


@pytest.mark.skipif(not have_real, reason="run `python -m stale_but_safe.env.real prepare` first")
def test_real_trace_scaled_to_load():
    topo, trace = real.load("abilene")
    assert trace.shape[1] == topo.D and topo.cap.min() > 0
    env = TEEnv(topo, EnvCfg(load=0.8, horizon=20, trace=trace), 0)
    us = []
    for _ in range(20):
        us.append(ecmp_max_util(topo, env.peek_demand()))
        from stale_but_safe.env.traffic import ecmp_splits
        env.step(ecmp_splits(topo))
    assert 0.2 < np.median(us) < 2.0          # same order as the requested median load


def test_trace_replay_is_deterministic_and_real():
    topo = synthetic("ba", 8, 1)
    trace = np.abs(np.random.default_rng(0).normal(size=(300, topo.D)))
    a = TEEnv(topo, EnvCfg(load=0.7, horizon=10, trace=trace), 5)
    b = TEEnv(topo, EnvCfg(load=0.7, horizon=10, trace=trace), 5)
    assert np.allclose(a.peek_demand(), b.peek_demand())
    row = a.peek_demand() / a.traffic.scale
    assert any(np.allclose(row, r, atol=1e-9) for r in trace)


def test_audit_runs_and_gives_verdict():
    topo = synthetic("ba", 8, 3)
    out = A.audit(topo, None, TelemetryCfg(5, 1, 0.05, 0.02), 0.8, 20, 1, None)
    assert out["verdict"]["action"] and set(out["results"]) == {"ideal", "snmp", "degraded", "yours"}
    assert "<svg" in A.to_html(out)


def test_audit_insufficient_evidence_with_few_windows():
    topo = synthetic("ba", 8, 3)
    out = A.audit(topo, None, TelemetryCfg(5, 1, 0.05, 0.02), 0.8, 20, 2, None)
    assert out["verdict"]["action"].startswith("Insufficient evidence")
    lo, hi = out["results"]["ideal"]["tuned"]["gain_ci_pct"]
    assert lo <= hi and out["results"]["ideal"]["tuned"]["windows"] == 2


def test_audit_flags_out_of_range_telemetry():
    assert A.ood_notes(TelemetryCfg(30, 1, 0.05, 0.02)) and not A.ood_notes(TelemetryCfg(5, 3, 0.2, 0.1))


def test_rate_limiter_reduces_updates_and_keeps_valid_splits():
    from stale_but_safe.baselines.heuristics import TunedHeuristic
    from stale_but_safe.baselines.ratelimit import RateLimited
    from stale_but_safe.eval.runner import rollout
    topo = synthetic("ba", 10, 5)
    cfg = EnvCfg(load=0.8, horizon=30)
    counts = {}
    for h in (0.0, 0.5):
        pol = RateLimited(TunedHeuristic(), h)
        rollout(topo, cfg, pol, 0)
        counts[h] = sum(pol.updates)
        assert (pol.installed >= -1e-9).all() and np.allclose(pol.installed.sum(1), 1.0, atol=1e-6)
    assert counts[0.5] <= counts[0.0]
