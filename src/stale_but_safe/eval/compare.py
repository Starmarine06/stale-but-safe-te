"""Compare policies over topologies x telemetry profiles on matched seeds."""
from __future__ import annotations

import numpy as np

from ..env.te_env import EnvCfg
from ..env.telemetry import PROFILES, TelemetryCfg
from .runner import rollout


def episode_stats(topo, cfg, policy, seed) -> dict:
    r = rollout(topo, cfg, policy, seed)
    return {
        "mu": float(r["max_util"].mean()), "ecmp_mu": float(r["ecmp_max_util"].mean()),
        "loss": float(r["loss"].mean()), "ecmp_loss": float(r["ecmp_loss"].mean()),
        "churn": float(r["churn"].mean()),
        "regret": r["regret"],
    }


def score(s: dict) -> float:
    """Composite (lower better): max-util ratio vs ECMP + 5*loss, normalised so
    ECMP = 1 + 5*ecmp_loss. We report the paired difference vs ECMP below."""
    return s["mu"] / s["ecmp_mu"] + 5 * s["loss"]


def ecmp_score(s: dict) -> float:
    return 1.0 + 5 * s["ecmp_loss"]


def table(topos, policies: dict, profiles: dict[str, TelemetryCfg], seeds, make_cfg=None,
          load=0.8, horizon=100):
    """Returns {(profile, policy): list of per-episode dicts with 'delta' = score - ecmp_score
    (negative = better than ECMP) and 'regret' arrays}."""
    out: dict = {}
    for pn, tele in profiles.items():
        for pol_name, pol_fn in policies.items():
            rows = []
            for t in topos:
                cfg = make_cfg(tele) if make_cfg else EnvCfg(load=load, horizon=horizon, telemetry=tele)
                pol = pol_fn(pn, t)
                for s in seeds:
                    st = episode_stats(t, cfg, pol, s)
                    rows.append({"topo": t.name, "seed": s, "delta": score(st) - ecmp_score(st),
                                 "gain": 1 - st["mu"] / st["ecmp_mu"], "loss": st["loss"],
                                 "churn": st["churn"], "regret": st["regret"]})
            out[(pn, pol_name)] = rows
    return out


def summarize(res) -> list[tuple]:
    rows = []
    for (pn, name), r in res.items():
        reg = np.concatenate([x["regret"] for x in r])
        rows.append((pn, name, float(np.mean([x["delta"] for x in r])),
                     float(np.mean([x["gain"] for x in r])),
                     float(np.quantile(reg, 0.99)), float(np.mean([x["loss"] for x in r])),
                     float(np.mean([x["churn"] for x in r]))))
    return rows


def print_table(rows):
    print(f"{'profile':9s} {'policy':14s} {'dScore':>8s} {'gain%':>7s} {'p99reg':>8s} {'loss':>7s} {'churn':>7s}")
    for pn, name, d, g, p99, loss, ch in rows:
        print(f"{pn:9s} {name:14s} {d:8.3f} {100*g:7.1f} {p99:8.3f} {loss:7.3f} {ch:7.3f}")
