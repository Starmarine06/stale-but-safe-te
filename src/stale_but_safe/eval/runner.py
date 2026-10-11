"""Roll out policies on paired seeds and compute regret vs ECMP."""
from __future__ import annotations

import numpy as np

from ..env.te_env import EnvCfg, TEEnv


def rollout(topo, cfg: EnvCfg, policy, seed: int) -> dict:
    env = TEEnv(topo, cfg, seed)
    obs = env.reset(seed)
    policy.reset(env)
    keys = ["max_util", "util_var", "loss", "delay", "churn", "ecmp_max_util", "ecmp_loss"]
    rec = {k: [] for k in keys}
    done = False
    while not done:
        obs, r, done, info = env.step(policy.act(obs, env))
        for k in keys:
            rec[k].append(info[k])
    out = {k: np.array(v) for k, v in rec.items()}
    out["regret"] = out["max_util"] - out["ecmp_max_util"]  # >0 means worse than ECMP
    return out


def evaluate(topo, cfg: EnvCfg, policy, seeds) -> dict:
    """Aggregate over seeds. Same seed => identical traffic across policies."""
    runs = [rollout(topo, cfg, policy, s) for s in seeds]
    cat = {k: np.concatenate([r[k] for r in runs]) for k in runs[0]}
    return {
        "mean_max_util": float(cat["max_util"].mean()),
        "mean_ecmp_max_util": float(cat["ecmp_max_util"].mean()),
        "improvement_vs_ecmp": float(1 - cat["max_util"].mean() / cat["ecmp_max_util"].mean()),
        "mean_loss": float(cat["loss"].mean()),
        "regret_mean": float(cat["regret"].mean()),
        "regret_p99": float(np.quantile(cat["regret"], 0.99)),
        "regret_cvar95": float(np.sort(cat["regret"])[-max(1, len(cat["regret"]) // 20):].mean()),
        "churn": float(cat["churn"].mean()),
        "raw": cat,
    }
