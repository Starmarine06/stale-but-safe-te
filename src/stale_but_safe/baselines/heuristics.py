"""Non-learning baselines. Every policy has `reset(env)` and `act(obs, env)`;
only `LPOracle` is allowed to use `env.peek_demand()` / true state."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from ..env.te_env import TEEnv
from ..env.traffic import ecmp_splits


class Policy:
    name = "policy"

    def reset(self, env: TEEnv):
        self.env, self.topo = env, env.topo

    def act(self, obs: dict, env: TEEnv) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class ECMP(Policy):
    name = "ecmp"

    def act(self, obs, env):
        return ecmp_splits(self.topo, self.topo.valid_paths(obs["link_up"]))


class SPF(Policy):
    """Single min-hop path (the first candidate that is still valid)."""
    name = "spf"

    def act(self, obs, env):
        valid = self.topo.valid_paths(obs["link_up"])
        out = np.zeros(valid.shape)
        first = np.argmax(valid, axis=1)
        out[np.arange(len(first)), first] = valid.any(1)
        return out


def _path_cost(topo, util, valid, agg="max"):
    """(D, k) path cost from per-link util; invalid paths = inf."""
    u = np.asarray(util)
    if agg == "max":
        cost = np.zeros((topo.D, topo.k))
        for d in range(topo.D):
            for p, ls in enumerate(topo.path_links[d]):
                cost[d, p] = u[ls].max()
    else:
        cost = (topo.inc.T @ u).reshape(topo.D, topo.k)
    return np.where(valid, cost, np.inf)


class Greedy(Policy):
    """Each demand goes entirely on the path with the lowest estimated bottleneck."""
    name = "greedy"

    def act(self, obs, env):
        valid = self.topo.valid_paths(obs["link_up"])
        cost = _path_cost(self.topo, obs["util"], valid)
        out = np.zeros(cost.shape)
        best = np.argmin(cost, axis=1)
        out[np.arange(len(best)), best] = np.isfinite(cost.min(1))
        return out


class TunedHeuristic(Policy):
    """EWMA-smoothed bottleneck softmin with hysteresis and damped updates.

    params: ewma (smoothing of util estimate), tau (softmin temperature),
    hyst (relative cost gain required to move a demand), step (blend toward
    new split), age_decay (shrink stale readings toward the mean).
    """
    name = "tuned"
    DEFAULT = dict(ewma=0.5, tau=0.1, hyst=0.05, step=0.5, age_decay=0.0)

    def __init__(self, **params):
        self.p = {**self.DEFAULT, **params}

    def reset(self, env):
        super().reset(env)
        self.smooth = None
        self.prev = ecmp_splits(self.topo)

    def act(self, obs, env):
        p, topo = self.p, self.topo
        u = np.asarray(obs["util"], float)
        if p["age_decay"] > 0:  # trust old readings less
            w = np.exp(-p["age_decay"] * obs["util_age"])
            u = w * u + (1 - w) * u.mean()
        self.smooth = u if self.smooth is None else p["ewma"] * u + (1 - p["ewma"]) * self.smooth
        valid = topo.valid_paths(obs["link_up"])
        cost = _path_cost(topo, self.smooth, valid)
        finite = np.where(np.isfinite(cost), cost, 1e9)
        z = -(finite - finite.min(1, keepdims=True)) / p["tau"]
        w = np.where(valid, np.exp(np.clip(z, -50, 0)), 0.0)
        target = w / np.maximum(w.sum(1, keepdims=True), 1e-12)
        # hysteresis: only move demands whose current split is clearly worse
        cur_cost = (self.prev * np.where(valid, cost, 0)).sum(1)
        gain = cur_cost - (target * np.where(valid, cost, 0)).sum(1)
        move = (gain > p["hyst"])[:, None]
        new = np.where(move, (1 - p["step"]) * self.prev + p["step"] * target, self.prev)
        new = new * valid
        s = new.sum(1, keepdims=True)
        new = np.where(s > 1e-9, new / np.maximum(s, 1e-12), ecmp_splits(topo, valid))
        self.prev = new
        return new


def solve_min_maxutil(topo, demand, valid, cap_scale=None):
    """LP: minimise z s.t. link load <= z * cap. Returns (splits, z)."""
    D, k, E = topo.D, topo.k, topo.E
    nv = D * k
    c = np.zeros(nv + 1)
    c[-1] = 1.0
    A_ub = np.hstack([topo.inc * np.tile(np.repeat(demand, k), 1)[None, :],
                      -topo.cap[:, None]])
    b_ub = np.zeros(E)
    vmask = valid.ravel()
    A_eq = np.zeros((D, nv + 1))
    for d in range(D):
        A_eq[d, d * k:(d + 1) * k] = 1.0
    has = valid.any(1)
    b_eq = has.astype(float)
    bounds = [(0, 1 if vmask[i] else 0) for i in range(nv)] + [(0, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        return ecmp_splits(topo, valid), float("nan")
    return res.x[:nv].reshape(D, k), float(res.x[-1])


class LPOracle(Policy):
    """Perfect-information min-max-utilisation multi-commodity flow (upper bound)."""
    name = "oracle"

    def act(self, obs, env):
        valid = self.topo.valid_paths(env.link_up)
        s, _ = solve_min_maxutil(self.topo, env.peek_demand(), valid)
        return s


# ------------------------------------------------------------------- tuning
def tune_heuristic(make_env, eval_fn, budget: int = 40, seed: int = 0):
    """Random search over TunedHeuristic params. `eval_fn(policy, make_env)`
    returns a scalar to minimise (e.g. mean max-util under the target telemetry).
    Use the same `budget` as the RL agent's hyperparameter budget."""
    rng = np.random.default_rng(seed)
    best, best_score = dict(TunedHeuristic.DEFAULT), None
    for i in range(budget):
        cand = dict(ewma=rng.uniform(0.1, 1.0), tau=10 ** rng.uniform(-2, 0),
                    hyst=rng.uniform(0, 0.2), step=rng.uniform(0.1, 1.0),
                    age_decay=rng.choice([0.0, 0.1, 0.3, 0.6]))
        if i == 0:
            cand = dict(TunedHeuristic.DEFAULT)
        score = eval_fn(TunedHeuristic(**cand), make_env)
        if best_score is None or score < best_score:
            best, best_score = cand, score
    return best, best_score
