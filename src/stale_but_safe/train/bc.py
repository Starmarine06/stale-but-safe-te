"""Privileged-teacher distillation (DAgger-style).

Teacher = two-phase LP oracle on the TRUE state. The student sees only the impaired
observation (history + age). Telemetry impairments are randomised per episode for
`--mode random`; `--mode ideal` trains on perfect telemetry (what most papers do).
Rollouts use a student/teacher mixture whose teacher share decays over iterations.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import numpy as np
import torch

from ..baselines.heuristics import solve_min_maxutil
from ..env.te_env import TEEnv
from ..models.features import Featurizer
from ..models.learned import ModelCfg, build, graph_static, save
from .scenarios import N_TRAIN_TOPOS, sample_cfg, train_topology

torch.set_num_threads(1)


def episode(args):
    """Worker: roll out one episode and return (topo_id, arrays) with teacher labels."""
    topo_id, seed, mode, mcfg, state, beta = args
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    topo = train_topology(topo_id)
    cfg = sample_cfg(rng, mode)
    env = TEEnv(topo, cfg, seed)
    obs = env.reset(seed)
    mc = ModelCfg(**mcfg)
    model = None
    if state is not None:
        model = build(mc)
        model.load_state_dict(state)
        model.eval()
    feat = Featurizer(topo, mc.H, mc.use_age, graph_static(topo))
    XL, XP, V, T, W = [], [], [], [], []
    done = False
    while not done:
        xl, xp, valid = feat(obs)
        dem = env.peek_demand()
        teach, _ = solve_min_maxutil(topo, dem, topo.valid_paths(env.link_up), two_phase=True)
        w = 0.2 + 0.8 * dem / max(dem.mean(), 1e-9)
        XL.append(xl); XP.append(xp); V.append(valid); T.append(teach.astype(np.float32))
        W.append(np.clip(w, 0, 5).astype(np.float32))
        if model is None or rng.random() < beta:
            act = teach
        else:
            with torch.no_grad():
                lg, _ = model(torch.from_numpy(xl)[None], torch.from_numpy(xp)[None],
                              torch.from_numpy(valid)[None], feat.G)
            act = torch.softmax(lg[0], -1).numpy()
        feat.note_action(act)
        obs, _, done, _ = env.step(act)
    f16 = lambda a: np.stack(a).astype(np.float16)
    return topo_id, (f16(XL), f16(XP), np.stack(V), f16(T), f16(W))


def fit(model, data, epochs, lr, bs=32):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    items = [(tid, *[torch.from_numpy(np.concatenate(c)) for c in zip(*chunks)])
             for tid, chunks in data.items() if chunks]
    tot, n = 0.0, 0
    for _ in range(epochs):
        for tid, XL, XP, V, T, W in items:
            G = graph_static(train_topology(tid))
            perm = torch.randperm(len(XL))
            for i in range(0, len(perm), bs):
                j = perm[i:i + bs]
                lg, _ = model(XL[j].float(), XP[j].float(), V[j], G)
                logp = torch.log_softmax(lg, -1).clamp(min=-30)
                ce = -(T[j].float() * logp * V[j]).sum(-1)
                w = W[j].float()
                loss = (w * ce).sum() / w.sum()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += loss.item()
                n += 1
    model.eval()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="random", choices=["random", "ideal"])
    ap.add_argument("--H", type=int, default=3)
    ap.add_argument("--no-age", action="store_true")
    ap.add_argument("--no-gnn", action="store_true")
    ap.add_argument("--hid", type=int, default=64)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--eps-per-iter", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-samples", type=int, default=14000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    mc = ModelCfg(H=a.H, use_age=not a.no_age, use_gnn=not a.no_gnn, hid=a.hid)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    model = build(mc)
    mcd = dict(mc.__dict__)
    data: dict = {}
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for it in range(a.iters):
            beta = 1.0 if it == 0 else max(0.0, 0.5 - 0.15 * (it - 1))
            state = None if it == 0 else {k: v.clone() for k, v in model.state_dict().items()}
            jobs = [(int(rng.integers(N_TRAIN_TOPOS)), int(rng.integers(1 << 30)), a.mode,
                     mcd, state, beta) for _ in range(a.eps_per_iter)]
            for tid, chunk in pool.imap_unordered(episode, jobs):
                data.setdefault(tid, []).append(chunk)
            # replay cap: drop the oldest chunks beyond max_samples (keeps RAM bounded)
            order = [(tid, i) for tid, ch in data.items() for i in range(len(ch))]
            total = sum(len(data[t][i][0]) for t, i in order)
            while total > a.max_samples:
                t_big = max(data, key=lambda t: len(data[t]))
                total -= len(data[t_big].pop(0)[0])
            loss = fit(model, data, a.epochs * (2 if it == 0 else 1), a.lr)
            n = sum(len(c[0]) for ch in data.values() for c in ch)
            print(f"iter {it} beta={beta:.2f} samples={n} loss={loss:.4f} t={time.time()-t0:.0f}s", flush=True)
    save(model, mc, a.out, extra={"args": vars(a)})


if __name__ == "__main__":
    main()
