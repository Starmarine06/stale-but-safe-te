"""Training-distribution samplers (synthetic topologies, traffic, telemetry)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..env.te_env import EnvCfg
from ..env.telemetry import PROFILES, TelemetryCfg
from ..env.topology import synthetic

KINDS = ("ba", "er", "geo")
N_TRAIN_TOPOS = 40


@lru_cache(maxsize=None)
def train_topology(i: int):
    """Training topologies: seeds >= 1000, n in [10, 20]. Never used for tuning/test."""
    rng = np.random.default_rng(1000 + i)
    return synthetic(KINDS[i % 3], int(rng.integers(10, 21)), seed=1000 + i)


def sample_telemetry(rng, mode: str) -> TelemetryCfg:
    if mode == "ideal":
        return PROFILES["ideal"]
    if rng.random() < 0.3:
        return PROFILES[rng.choice(list(PROFILES))]
    return TelemetryCfg(P=int(rng.choice([1, 2, 3, 5, 10])), d=int(rng.choice([0, 1, 2, 3, 5])),
                        sigma=float(rng.uniform(0, 0.3)), p=float(rng.choice([0, 0.05, 0.1, 0.2])))


def sample_cfg(rng, mode: str, horizon: int = 100) -> EnvCfg:
    traffic = {}
    r = rng.random()
    if r < 0.2:
        traffic["hotspot_prob"] = 0.1
    elif r < 0.3:
        traffic["shift_at"] = horizon // 2
    elif r < 0.4:
        traffic.update(flash_at=horizon // 3, flash_len=20)
    fail = int(horizon * 0.4) if rng.random() < 0.15 else None
    return EnvCfg(load=float(rng.choice([0.5, 0.8, 1.1])), horizon=horizon,
                  telemetry=sample_telemetry(rng, mode), L=int(rng.choice([0, 0, 2])),
                  traffic=traffic, fail_at=fail)
