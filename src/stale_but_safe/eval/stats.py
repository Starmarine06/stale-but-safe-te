"""Statistics per docs/eval_protocol.md: bootstrap CIs, IQM, paired Wilcoxon + Holm,
CVaR and performance profiles (rliable-style)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def bootstrap_ci(x, stat=np.mean, n=10000, alpha=0.05, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    s = np.array([stat(x[i]) for i in idx]) if stat is not np.mean else x[idx].mean(1)
    return float(stat(x)), float(np.quantile(s, alpha / 2)), float(np.quantile(s, 1 - alpha / 2))


def iqm(x):
    return float(stats.trim_mean(np.asarray(x, float), 0.25))


def cvar(x, q=0.95):
    x = np.sort(np.asarray(x, float))
    return float(x[int(np.floor(q * len(x))):].mean())


def wilcoxon_holm(pairs: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
    """pairs: name -> (a, b) paired samples. Returns Holm-adjusted p-values (two-sided)."""
    names, ps = [], []
    for k, (a, b) in pairs.items():
        d = np.asarray(a, float) - np.asarray(b, float)
        p = 1.0 if np.allclose(d, 0) else float(stats.wilcoxon(d).pvalue)
        names.append(k); ps.append(p)
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, run)
    return dict(zip(names, adj.tolist()))


def performance_profile(scores: dict[str, np.ndarray], taus: np.ndarray):
    """Fraction of runs with score < tau (lower score = better) for each method."""
    return {k: np.array([(np.asarray(v) < t).mean() for t in taus]) for k, v in scores.items()}
