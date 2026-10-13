"""Action-rate limiter: keep a demand's installed split unless the proposal moves it by more than `hyst`
(half the L1 distance). Counts per-interval updates (= flow/group modifications a controller would send)."""
from __future__ import annotations

import numpy as np

from .heuristics import Policy
from ..env.traffic import ecmp_splits


class RateLimited(Policy):
    def __init__(self, inner: Policy, hyst: float):
        self.inner, self.hyst = inner, hyst
        self.name = f"{inner.name}@h{hyst:g}"

    def reset(self, env):
        super().reset(env)
        self.inner.reset(env)
        self.installed = ecmp_splits(self.topo)
        self.updates = []

    def act(self, obs, env):
        prop = np.asarray(self.inner.act(obs, env), float)
        valid = self.topo.valid_paths(obs["link_up"])
        move = np.abs(prop - self.installed).sum(1) / 2
        broken = (self.installed * ~valid).sum(1) > 1e-9           # installed path now down: must change
        upd = (move > self.hyst) | broken
        self.installed = np.where(upd[:, None], prop, self.installed)
        self.updates.append(int(upd.sum()))
        return self.installed.copy()
