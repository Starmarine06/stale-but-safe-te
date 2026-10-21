"""Three-layer runtime safety guard wrapping any policy; fallback = ECMP.

L1 hard rules   : stale/missing telemetry, malformed action, inference deadline.
L2 pre-action   : fluid-model check of the proposal vs ECMP on the *estimated* state;
                  reject if predicted max-util > ECMP * (1 + eps).   (eps = trade-off knob)
L3 monitor      : measured max-util vs the ECMP counterfactual predicted from the same
                  telemetry sample; if the policy is worse by > tau for k consecutive
                  intervals, latch fallback with exponential backoff.

The guarantee is empirical (p99 / CVaR regret vs ECMP), not a formal proof.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from ..baselines.heuristics import Policy
from ..env.traffic import ecmp_splits


@dataclass
class GuardCfg:
    eps: float = 0.0            # L2 tolerance over ECMP's predicted max-util
    max_mean_age: float = 8.0   # L1: mean telemetry age (intervals) above this -> ECMP
    deadline_s: float = 0.5     # L1: inference deadline
    layers: tuple = (True, True, True)
    tau: float = 0.05           # L3: tolerated measured excess over ECMP counterfactual
    k_consec: int = 3
    window: int = 5
    backoff0: int = 4
    backoff_max: int = 64


def _pred_max_util(topo, demand, splits, link_up):
    load = topo.inc @ (demand[:, None] * splits).ravel()
    u = load / topo.cap
    return float(u[link_up].max()) if link_up.any() else 0.0


class Guard(Policy):
    def __init__(self, policy: Policy, cfg: GuardCfg | None = None):
        self.policy, self.cfg = policy, cfg or GuardCfg()
        self.name = f"{policy.name}+guard"

    def reset(self, env):
        super().reset(env)
        self.policy.reset(env)
        self.excess = deque(maxlen=self.cfg.window)
        self.bad_run = 0
        self.fallback_left = 0
        self.backoff = self.cfg.backoff0
        self.good_since = 0
        self.stats = {"steps": 0, "fallback": 0, "L1": 0, "L2": 0, "L3": 0}
        self.trace = []     # per-call diagnostics: [layer fired or 0, predicted proposal mu, predicted ECMP mu, mean excess, measured max util, counterfactual ECMP mu]
        self._d = [0, np.nan, np.nan, np.nan, np.nan, np.nan]

    def _fallback(self, obs, why):
        self.stats["fallback"] += 1
        self.stats[why] += 1
        self._d[0] = int(why[1])
        return ecmp_splits(self.topo, self.topo.valid_paths(obs["link_up"]))

    def act(self, obs, env):
        self._d = [0, np.nan, np.nan, np.nan, np.nan, np.nan]
        out = self._act(obs, env)
        self.trace.append(self._d)
        return out

    def _act(self, obs, env):
        c, topo = self.cfg, self.topo
        self.stats["steps"] += 1
        up = obs["link_up"]
        valid = topo.valid_paths(up)
        ecmp = ecmp_splits(topo, valid)
        dem = np.asarray(obs["demand"], float)

        # L3 bookkeeping runs every step (also in fallback) so we can re-enable.
        if c.layers[2]:
            cf = _pred_max_util(topo, dem, ecmp, up)
            meas = float(np.asarray(obs["util"])[up].max()) if up.any() else 0.0
            self.excess.append(meas - cf)
            self._d[3], self._d[4], self._d[5] = float(np.mean(self.excess)), meas, cf
        if self.fallback_left > 0:
            self.fallback_left -= 1
            self.stats["fallback"] += 1
            self.stats["L3"] += 1
            self._d[0] = 3
            return ecmp

        # L1 hard rules (pre-inference)
        if c.layers[0] and (np.mean(obs["util_age"]) > c.max_mean_age
                            or not np.isfinite(obs["util"]).all()):
            return self._fallback(obs, "L1")

        t0 = time.perf_counter()
        proposal = np.asarray(self.policy.act(obs, env), float)
        dt = time.perf_counter() - t0
        if c.layers[0]:
            bad = (proposal.shape != ecmp.shape or not np.isfinite(proposal).all()
                   or (proposal < -1e-9).any() or (proposal[~valid] > 1e-9).any()
                   or not np.allclose(proposal.sum(1), valid.any(1), atol=1e-3)
                   or dt > c.deadline_s)
            if bad:
                return self._fallback(obs, "L1")

        # L2 pre-action check on the estimated state
        if c.layers[1]:
            p = _pred_max_util(topo, dem, proposal, up)
            e = _pred_max_util(topo, dem, ecmp, up)
            self._d[1], self._d[2] = p, e
            if p > e * (1 + c.eps) + 1e-9:
                return self._fallback(obs, "L2")

        # L3 runtime monitor: latch fallback if the policy underperforms
        if c.layers[2] and len(self.excess) == c.window:
            if float(np.mean(self.excess)) > c.tau:
                self.bad_run += 1
            else:
                self.bad_run = 0
                self.good_since += 1
                if self.good_since > 3 * c.window:
                    self.backoff = c.backoff0
            if self.bad_run >= c.k_consec:
                self.fallback_left = self.backoff
                self.backoff = min(self.backoff * 2, c.backoff_max)
                self.bad_run = 0
                self.good_since = 0
                self.excess.clear()
                return self._fallback(obs, "L3")
        return proposal
