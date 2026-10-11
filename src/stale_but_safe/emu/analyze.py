"""Compare emulation traces with the simulator on the same scenario (RQ4 sim->emu gap)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from ..env.telemetry import PROFILES
from . import scenario as S
from .run_emulation import tag_of


def load_emulation(run_dir: Path, tag: str, sc: dict) -> dict:
    ctrl = json.loads((run_dir / tag / f"ctrl_{tag}.json").read_text())
    rows = ctrl["rows"]
    steps = sc["steps"]
    # interval i is measured at tick i+1
    mu = np.full(steps, np.nan)
    for r in rows:
        i = r["interval"]
        if 0 <= i < steps:
            mu[i] = r["true_max_util"]
    sent = np.zeros(steps)
    got = np.zeros(steps)
    wire = S.WIRE_FACTOR
    for d in sc["active"]:
        rx = run_dir / tag / f"rx_{d}.json"
        if rx.exists():
            g = json.loads(rx.read_text())[:steps]
            got += np.array(g) * 8 / 1e6
        sent += np.array([sc["schedule"][i][d] for i in range(steps)]) * sc["delta"]
    sent_mbit = sent * sc["mbps_per_unit"] / wire
    loss = 1 - got / np.maximum(sent_mbit, 1e-9)
    return {"max_util": mu, "loss": np.clip(loss, -0.5, 1), "rows": rows, "guard": ctrl.get("guard", {}),
            "latency_ms": np.array([r["latency_ms"] for r in rows if r["latency_ms"] > 0]),
            "changed": np.array([r["changed"] for r in rows])}


def sim_trace(sc: dict, spec: str, profile: str, make_policy) -> dict:
    from ..env.te_env import TEEnv
    topo = S.topo_from_dict(sc["topo"])
    cfg = S.sim_cfg(sc, PROFILES[profile])
    env = TEEnv(topo, cfg, sc["seed"])
    obs = env.reset(sc["seed"])
    pol = make_policy(spec, profile)
    pol.reset(env)
    mu, ecmp_mu, loss = [], [], []
    done = False
    while not done:
        obs, _, done, info = env.step(pol.act(obs, env))
        mu.append(info["max_util"]); ecmp_mu.append(info["ecmp_max_util"]); loss.append(info["loss"])
    return {"max_util": np.array(mu), "ecmp_max_util": np.array(ecmp_mu), "loss": np.array(loss)}


def compare(run_dir: Path, sc: dict, runs: list[tuple[str, str]], make_policy, skip=3) -> list[dict]:
    out = []
    for spec, prof in runs:
        tag = tag_of(spec, prof)
        try:
            em = load_emulation(run_dir, tag, sc)
        except FileNotFoundError:
            continue
        sm = sim_trace(sc, spec, prof, make_policy)
        a, b = em["max_util"][skip:], sm["max_util"][skip:]
        ok = np.isfinite(a)
        out.append({
            "spec": spec, "profile": prof,
            "emu_mu": float(np.nanmean(a)), "sim_mu": float(b.mean()),
            "pearson": float(stats.pearsonr(a[ok], b[ok])[0]) if ok.sum() > 3 else float("nan"),
            "spearman": float(stats.spearmanr(a[ok], b[ok])[0]) if ok.sum() > 3 else float("nan"),
            "emu_loss": float(np.nanmean(em["loss"][skip:])), "sim_loss": float(sm["loss"][skip:].mean()),
            "latency_ms": float(np.mean(em["latency_ms"])) if len(em["latency_ms"]) else float("nan"),
            "flowmods_per_min": float(em["changed"].sum() / (sc["steps"] * sc["delta"] / 60)),
            "guard": em["guard"],
        })
    return out
