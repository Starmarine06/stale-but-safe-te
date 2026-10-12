"""Synthetic traffic-matrix generator: gravity base, diurnal cycle, bursts,
hotspots, an optional abrupt pattern shift and a flash crowd."""
from __future__ import annotations

import numpy as np

from .topology import Topology


def ecmp_splits(topo: Topology, valid: np.ndarray | None = None) -> np.ndarray:
    """Equal split over the min-hop valid paths of each demand. Rows with no
    valid path are all zero."""
    valid = topo.path_mask if valid is None else valid
    h = np.where(valid, topo.hops, 10**6)
    best = h.min(axis=1, keepdims=True)
    w = (valid & (h == best) & (best < 10**6)).astype(float)
    s = w.sum(axis=1, keepdims=True)
    return np.divide(w, s, out=np.zeros_like(w), where=s > 0)


def ecmp_max_util(topo: Topology, demand: np.ndarray) -> float:
    x = (demand[:, None] * ecmp_splits(topo)).ravel()
    return float(((topo.inc @ x) / topo.cap).max())


class TrafficGen:
    def __init__(self, topo: Topology, rng: np.random.Generator, load: float = 0.5,
                 diurnal_amp: float = 0.3, diurnal_period: int = 200,
                 burst_sigma: float = 0.25, hotspot_prob: float = 0.02,
                 hotspot_mult: float = 4.0, shift_at: int | None = None,
                 flash_at: int | None = None, flash_len: int = 20,
                 flash_mult: float = 5.0, active=None):
        """`load` = ECMP max link utilisation of the *base* matrix."""
        self.topo, self.rng = topo, rng
        self.amp, self.period, self.bsig = diurnal_amp, diurnal_period, burst_sigma
        self.hp, self.hm = hotspot_prob, hotspot_mult
        self.shift_at, self.flash_at = shift_at, flash_at
        self.flash_len, self.flash_mult = flash_len, flash_mult
        self.phase = rng.uniform(0, 2 * np.pi)
        self.load, self.t, self._hot = load, 0, np.ones(topo.D)
        self.flash_node = int(rng.integers(topo.n))
        self.active = None if active is None else np.asarray(active, bool)
        self._set_base()

    def _set_base(self):
        mass = self.rng.lognormal(0.0, 0.8, self.topo.n)
        s, t = np.array(self.topo.demands).T
        base = mass[s] * mass[t]
        if self.active is not None:
            base = base * self.active
        self.base = base * self.load / ecmp_max_util(self.topo, base)

    def next(self) -> np.ndarray:
        t = self.t
        if self.shift_at is not None and t == self.shift_at:
            self._set_base()
        d = self.base * (1 + self.amp * np.sin(2 * np.pi * t / self.period + self.phase))
        d = d * self.rng.lognormal(-self.bsig ** 2 / 2, self.bsig, self.topo.D)
        # hotspots: a demand spikes for a few steps
        self._hot = 1 + (self._hot - 1) * 0.6
        spike = self.rng.random(self.topo.D) < self.hp
        self._hot[spike] = self.hm
        d = d * self._hot
        if self.flash_at is not None and self.flash_at <= t < self.flash_at + self.flash_len:
            dst = np.array([tt for _, tt in self.topo.demands]) == self.flash_node
            d = np.where(dst, d * self.flash_mult, d)
        self.t += 1
        return d
