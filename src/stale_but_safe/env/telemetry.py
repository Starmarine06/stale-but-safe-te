"""Telemetry impairment model: polling period P, observation delay d,
multiplicative noise sigma, per-entry drop probability p (stale value kept).

Optional realism extensions (all off by default, so seeded results are unchanged): `outlier` gross errors on a fraction of entries,
`round_drop` loss of a whole poll round, `jitter` poll-timing jitter."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TelemetryCfg:
    P: int = 1          # poll every P intervals
    d: int = 0          # reading age (intervals) at poll time
    sigma: float = 0.0  # log-normal multiplicative noise (mean-preserving)
    p: float = 0.0      # probability an entry is missing in a poll round
    outlier: float = 0.0     # probability an entry carries a gross error (reading scaled by U(0, 2))
    round_drop: float = 0.0  # probability a whole poll round is lost (all entries keep their stale value)
    jitter: float = 0.0      # probability a poll happens one interval early or late


PROFILES = {
    "ideal": TelemetryCfg(1, 0, 0.0, 0.0),
    "snmp": TelemetryCfg(5, 1, 0.05, 0.02),
    "rich": TelemetryCfg(1, 1, 0.03, 0.0),
    "degraded": TelemetryCfg(5, 3, 0.2, 0.1),
}


class Channel:
    """One impaired measurement stream of fixed shape."""

    def __init__(self, shape, cfg: TelemetryCfg, rng: np.random.Generator):
        self.cfg, self.rng = cfg, rng
        self.hist: deque = deque(maxlen=cfg.d + 1)
        self.value = np.zeros(shape)
        self.age = np.zeros(shape)
        self.t = 0
        self.next_poll = 0

    def push(self, true: np.ndarray):
        c = self.cfg
        self.hist.append(np.asarray(true, float).copy())
        first = self.t == 0
        due = self.t == self.next_poll if c.jitter > 0 else self.t % c.P == 0
        if due and c.jitter > 0:
            step = c.P + (int(self.rng.integers(-1, 2)) if self.rng.random() < c.jitter and c.P > 1 else 0)
            self.next_poll = self.t + max(1, step)
        if due:
            src = self.hist[0]  # oldest sample = d intervals old once warm
            noise = np.exp(c.sigma * self.rng.standard_normal(src.shape) - c.sigma ** 2 / 2) \
                if c.sigma > 0 else 1.0
            ok = np.ones(src.shape, bool) if first else self.rng.random(src.shape) >= c.p
            if c.outlier > 0:
                bad = self.rng.random(src.shape) < c.outlier
                noise = np.where(bad, self.rng.uniform(0.0, 2.0, src.shape), noise)
            if c.round_drop > 0 and not first and self.rng.random() < c.round_drop:
                ok = np.zeros(src.shape, bool)
            self.value = np.where(ok, src * noise, self.value)
            self.age = np.where(ok, float(c.d), self.age + 1)
        else:
            self.age = self.age + 1
        self.t += 1

    def observe(self) -> tuple[np.ndarray, np.ndarray]:
        """(value, age in intervals since the measured sample was true)."""
        return self.value.copy(), self.age.copy()
