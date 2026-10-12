import numpy as np
import pytest
import torch

from stale_but_safe.emu import scenario as S
from stale_but_safe.env.te_env import EnvCfg, TEEnv
from stale_but_safe.env.telemetry import PROFILES
from stale_but_safe.env.topology import synthetic
from stale_but_safe.env.traffic import ecmp_splits
from stale_but_safe.models.features import Featurizer, GraphStatic
from stale_but_safe.train.dfm import fluid_loss


@pytest.fixture(scope="module")
def topo():
    return synthetic("ba", 10, 7777)


def test_scenario_roundtrip_and_ports(topo):
    sc = S.make_scenario(topo, EnvCfg(load=0.7), seed=0, steps=12, active_n=8)
    t2 = S.topo_from_dict(sc["topo"])
    assert t2.E == topo.E and t2.D == topo.D
    assert np.array_equal(t2.inc, topo.inc) and np.array_equal(t2.path_mask, topo.path_mask)
    out = {int(k): v for k, v in sc["ports"]["out_port"].items()}
    # every switch has unique, >=2 ports for its links; both ends of a link are listed
    per_sw = {}
    for l, (a, _) in enumerate(topo.links):
        per_sw.setdefault(a, []).append(out[l])
    for a, ports in per_sw.items():
        assert len(set(ports)) == len(ports) and min(ports) >= 2


def test_scenario_schedule_matches_simulator(topo):
    sc = S.make_scenario(topo, EnvCfg(load=0.7), seed=3, steps=10, active_n=6)
    env = TEEnv(topo, S.sim_cfg(sc, PROFILES["ideal"]), 3)
    env.reset(3)
    for i in range(10):
        assert np.allclose(env.peek_demand(), sc["schedule"][i])
        env.step(ecmp_splits(topo))
    assert (np.array(sc["schedule"])[:, [d for d in range(topo.D) if d not in sc["active"]]] == 0).all()


def test_carried_obs_clips_at_one(topo):
    cfg = EnvCfg(load=1.5, horizon=5, carried_obs=True)
    env = TEEnv(topo, cfg, 0)
    obs = env.reset(0)
    for _ in range(4):
        obs, _, _, info = env.step(ecmp_splits(topo))
        assert obs["util"].max() <= 1.0 + 1e-9
    assert info["max_util"] > 1.0     # true (offered) utilisation is not clipped


def test_fluid_loss_matches_simulator(topo):
    env = TEEnv(topo, EnvCfg(load=1.3), 0)
    env.reset(0)
    dem = env.peek_demand()
    splits = ecmp_splits(topo)
    _, _, _, info = env.step(splits)
    G = GraphStatic(topo)
    cap = torch.tensor(topo.cap, dtype=torch.float32)
    sp = torch.tensor(splits, dtype=torch.float32)[None]
    d = torch.tensor(dem, dtype=torch.float32)[None]
    _, mu, loss = fluid_loss(sp, d, G, cap, torch.tensor([1.0]), sp, tau=1e-3, lam=0.0)
    assert mu == pytest.approx(info["max_util"], rel=1e-4)
    assert loss == pytest.approx(info["loss"], rel=1e-3, abs=1e-5)


def test_fluid_loss_is_differentiable(topo):
    G = GraphStatic(topo)
    cap = torch.tensor(topo.cap, dtype=torch.float32)
    logits = torch.zeros(1, topo.D, topo.k, requires_grad=True)
    sp = torch.softmax(logits, -1)
    dem = torch.rand(1, topo.D) * 0.2
    loss, _, _ = fluid_loss(sp, dem, G, cap, torch.tensor([1.0]), sp.detach())
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_no_model_features_are_hidden(topo):
    env = TEEnv(topo, EnvCfg(), 0)
    obs = env.reset(0)
    f1 = Featurizer(topo, 3, True, use_model=True)
    f0 = Featurizer(topo, 3, True, use_model=False)
    xl1, xp1, _ = f1(obs)
    xl0, xp0, _ = f0(obs)
    assert xl1.shape == xl0.shape and xp1.shape == xp0.shape
    assert np.abs(xp0[:, -2:]).sum() == 0 and np.abs(xp1[:, -2:]).sum() > 0
    assert np.abs(xl0[:, -4:-2]).sum() == 0
