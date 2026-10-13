"""Flow-level (fluid) traffic-engineering environment with impaired telemetry.

Decision loop (one interval):
    obs = env.observe()          # impaired view: utilisation, age, demand estimate
    splits = policy(obs)         # (D, k) split ratios over candidate paths
    env.step(splits)             # true state evolves; telemetry is pushed
The demand of the *next* interval is generated in advance so oracles can peek
(`env.peek_demand()`); learned/heuristic policies only see `observe()`.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .telemetry import Channel, TelemetryCfg
from .topology import Topology
from .traffic import TraceTraffic, TrafficGen, ecmp_splits


@dataclass
class EnvCfg:
    load: float = 0.5               # base ECMP max-util
    horizon: int = 200
    telemetry: TelemetryCfg = field(default_factory=TelemetryCfg)
    L: int = 0                      # rule-install latency (intervals)
    traffic: dict = field(default_factory=dict)  # TrafficGen overrides
    # reward weights
    alpha: float = 1.0              # max util
    beta: float = 0.0               # mean delay
    gamma: float = 5.0              # loss
    delta: float = 0.0              # util variance
    lam: float = 0.05               # churn
    carried_obs: bool = False       # telemetry shows carried (<=1) not offered utilisation, like real counters
    fail_at: int | None = None      # step at which random links fail
    n_fail: int = 1
    trace: object = None            # real demand trace (T, D); replaces the synthetic generator
    trace_stat: str = "median"      # load scaling reference: median | mean | p95 of the ECMP max-util over the trace


class TEEnv:
    def __init__(self, topo: Topology, cfg: EnvCfg | None = None, seed: int = 0):
        self.topo, self.cfg = topo, cfg or EnvCfg()
        self.seed = seed
        self.reset(seed)

    # ------------------------------------------------------------------ core
    def reset(self, seed: int | None = None):
        seed = self.seed if seed is None else seed
        self.seed = seed
        ss = np.random.SeedSequence(seed)
        r_traffic, r_tele, r_fail = (np.random.default_rng(s) for s in ss.spawn(3))
        c, topo = self.cfg, self.topo
        if c.trace is not None:
            self.traffic = TraceTraffic(topo, r_traffic, c.trace, load=c.load, horizon=c.horizon, stat=c.trace_stat)
        else:
            self.traffic = TrafficGen(topo, r_traffic, load=c.load, **c.traffic)
        self.fail_rng = r_fail
        self.link_up = np.ones(topo.E, bool)
        self.ch_util = Channel(topo.E, c.telemetry, r_tele)
        self.ch_dem = Channel(topo.D, c.telemetry, r_tele)
        self.t = 0
        self._next = self.traffic.next()
        ecmp = ecmp_splits(topo)
        self.applied = ecmp
        self.pending = deque([ecmp] * c.L)
        # prime telemetry with one ECMP interval so observe() is defined
        m = self._evaluate(self._next, ecmp)
        self.ch_util.push(np.minimum(m["util"], 1.0) if c.carried_obs else m["util"])
        self.ch_dem.push(self._next)
        self._maybe_fail()
        return self.observe()

    def _maybe_fail(self):
        """Port-status events are visible to the controller immediately."""
        c = self.cfg
        if c.fail_at is not None and self.t == c.fail_at:
            idx = self.fail_rng.choice(self.topo.E // 2, c.n_fail, replace=False) * 2
            self.link_up[idx] = False
            self.link_up[idx + 1] = False  # both directions

    def peek_demand(self) -> np.ndarray:
        """True demand of the interval about to be decided (oracle use only)."""
        return self._next

    def observe(self) -> dict:
        u, ua = self.ch_util.observe()
        d, da = self.ch_dem.observe()
        return {"util": u, "util_age": ua, "demand": d, "demand_age": da,
                "link_up": self.link_up.copy(), "t": self.t}

    def sanitize(self, splits: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
        """Project onto the simplex over valid paths; fall back to ECMP rows."""
        valid = self.topo.valid_paths(self.link_up) if valid is None else valid
        s = np.nan_to_num(np.asarray(splits, float), nan=0.0, posinf=0.0, neginf=0.0)
        s = np.clip(s, 0, None) * valid
        tot = s.sum(1, keepdims=True)
        fb = ecmp_splits(self.topo, valid)
        return np.where(tot > 1e-9, s / np.maximum(tot, 1e-12), fb)

    def _evaluate(self, demand: np.ndarray, splits: np.ndarray) -> dict:
        topo = self.topo
        valid = topo.valid_paths(self.link_up)
        has_path = valid.any(1)
        dem = np.where(has_path, demand, 0.0)
        x = (dem[:, None] * splits).ravel()
        load = topo.inc @ x
        util = load / topo.cap
        pass_frac = np.minimum(1.0, 1.0 / np.maximum(util, 1e-9))
        surv = np.exp(topo.inc.T @ np.log(pass_frac))
        tot = max(demand.sum(), 1e-12)
        delivered = float((x * surv).sum())
        loss = 1.0 - delivered / tot
        uc = np.minimum(util, 0.99)
        link_delay = topo.prop + 0.5 * uc / (1 - uc)
        path_delay = topo.inc.T @ link_delay
        mean_delay = float((x * path_delay).sum() / max(x.sum(), 1e-12))
        up = self.link_up
        return {"util": util, "max_util": float(util[up].max()) if up.any() else 0.0,
                "util_var": float(util[up].var()), "loss": loss, "delay": mean_delay}

    def step(self, splits: np.ndarray):
        c, topo = self.cfg, self.topo
        proposed = self.sanitize(splits)
        if c.L > 0:
            self.pending.append(proposed)
            eff = self.pending.popleft()
        else:
            eff = proposed
        eff = self.sanitize(eff)  # stale rules may point at newly failed links
        demand = self._next
        m = self._evaluate(demand, eff)
        base = self._evaluate(demand, self.sanitize(ecmp_splits(topo)))
        w = demand / max(demand.sum(), 1e-12)
        churn = float((w * np.abs(eff - self.applied).sum(1)).sum() / 2)
        self.applied = eff
        reward = -(c.alpha * m["max_util"] + c.beta * m["delay"] + c.gamma * m["loss"]
                   + c.delta * m["util_var"]) - c.lam * churn
        self.ch_util.push(np.minimum(m["util"], 1.0) if c.carried_obs else m["util"])
        self.ch_dem.push(demand)
        self.t += 1
        self._next = self.traffic.next()
        self._maybe_fail()
        info = {**{k: v for k, v in m.items() if k != "util"}, "churn": churn,
                "ecmp_max_util": base["max_util"], "ecmp_loss": base["loss"],
                "ecmp_delay": base["delay"], "util": m["util"]}
        return self.observe(), reward, self.t >= c.horizon, info
