"""Differentiable-fluid-model training (DOTE-style direct optimisation, under telemetry impairments).

The policy sees only the IMPAIRED observation; the loss is computed by pushing its splits through the
differentiable fluid model with the TRUE next-interval demand (available at training time only):

    L = softmax_tau(util)/max(mu_ecmp, .1) + 5 * fluid_loss + lam * churn

which is exactly the evaluation score. No teacher, no RL variance. Rollouts (DAgger-style) use the
current student with logit noise so the observation distribution matches what the student induces.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import numpy as np
import torch

from ..env.te_env import TEEnv
from ..env.traffic import ecmp_splits
from ..models.features import Featurizer
from ..models.learned import ModelCfg, build, graph_static, load, save
from .scenarios import N_TRAIN_TOPOS, sample_cfg, train_topology

torch.set_num_threads(1)
_CAP: dict = {}


def cap_t(topo):
    key = id(topo)
    if key not in _CAP:
        _CAP[key] = torch.tensor(topo.cap, dtype=torch.float32)
    return _CAP[key]


def fluid_loss(splits, dem, G, cap, ecmp_mu, prev, tau=0.03, lam=0.1, w_loss=5.0):
    """splits (B,D,k) dem (B,D) ecmp_mu (B,) prev (B,D,k) -> scalar loss and parts."""
    B, D, k = splits.shape
    x = (dem[:, :, None] * splits).reshape(B, D * k)
    util = x @ G.inc.t() / cap                                    # (B,E)
    smax = tau * torch.logsumexp(util / tau, -1)
    pas = torch.clamp(1.0 / util.clamp(min=1e-6), max=1.0)
    surv = torch.exp(torch.log(pas) @ G.inc)                       # (B,Dk)
    loss = 1.0 - (x * surv).sum(-1) / dem.sum(-1).clamp(min=1e-9)
    w = dem / dem.sum(-1, keepdim=True).clamp(min=1e-9)
    churn = (w * (splits - prev).abs().sum(-1)).sum(-1) / 2
    total = smax / ecmp_mu.clamp(min=0.1) + w_loss * loss + lam * churn
    return total.mean(), util.max(-1)[0].mean().item(), loss.mean().item()


def episode(args):
    topo_id, seed, mode, mcfg, state, sigma = args
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
    feat = Featurizer(topo, mc.H, mc.use_age, graph_static(topo), mc.use_model)
    gen = torch.Generator().manual_seed(int(seed))
    XL, XP, V, DM, PV, EM = [], [], [], [], [], []
    done = False
    while not done:
        xl, xp, valid = feat(obs)
        XL.append(xl); XP.append(xp); V.append(valid); DM.append(env.peek_demand().astype(np.float32))
        PV.append(feat.prev.astype(np.float32))
        t = lambda a: torch.from_numpy(a)[None]
        with torch.no_grad():
            if model is None:
                mu = torch.log(torch.from_numpy(ecmp_splits(topo, valid)).float().clamp(min=1e-3))[None]
            else:
                mu, _ = model(t(xl), t(xp), t(valid), feat.G)
            a = mu + sigma * torch.randn(mu.shape, generator=gen)
            act = torch.softmax(a[0].masked_fill(~t(valid)[0], -1e9), -1).numpy()
        feat.note_action(act)
        obs, _, done, info = env.step(act)
        EM.append(info["ecmp_max_util"])
    f16 = lambda a: np.stack(a).astype(np.float16)
    return topo_id, (f16(XL), f16(XP), np.stack(V), np.stack(DM), f16(PV), np.array(EM, np.float32))


def fit(model, data, epochs, lr, bs=32, lam=0.1):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    items = [(tid, *[torch.from_numpy(np.concatenate(c)) for c in zip(*chunks)])
             for tid, chunks in data.items() if chunks]
    tot, n, mus, losses = 0.0, 0, [], []
    for _ in range(epochs):
        for tid, XL, XP, V, DM, PV, EM in items:
            topo = train_topology(tid)
            G, cap = graph_static(topo), cap_t(topo)
            perm = torch.randperm(len(XL))
            for i in range(0, len(perm), bs):
                j = perm[i:i + bs]
                lg, _ = model(XL[j].float(), XP[j].float(), V[j], G)
                sp = torch.softmax(lg, -1)
                loss, mu, ls = fluid_loss(sp, DM[j], G, cap, EM[j], PV[j].float(), lam=lam)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += loss.item(); n += 1; mus.append(mu); losses.append(ls)
    model.eval()
    return tot / max(n, 1), float(np.mean(mus)), float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="random", choices=["random", "ideal"])
    ap.add_argument("--H", type=int, default=3)
    ap.add_argument("--no-age", action="store_true")
    ap.add_argument("--no-gnn", action="store_true")
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--hid", type=int, default=64)
    ap.add_argument("--init")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--eps-per-iter", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--max-samples", type=int, default=16000)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.init:
        model, mc = load(a.init)
    else:
        mc = ModelCfg(H=a.H, use_age=not a.no_age, use_gnn=not a.no_gnn, hid=a.hid,
                      use_model=not a.no_model)
        torch.manual_seed(a.seed)
        model = build(mc)
    rng = np.random.default_rng(a.seed + 7)
    data: dict = {}
    t0 = time.time()
    pool_cm = mp.get_context("spawn").Pool(a.workers) if a.workers > 0 else None
    run_eps = (lambda jobs: pool_cm.imap_unordered(episode, jobs)) if pool_cm else (lambda jobs: map(episode, jobs))
    try:
        for it in range(a.iters):
            state = None if (it == 0 and not a.init) else {k: v.clone() for k, v in model.state_dict().items()}
            jobs = [(int(rng.integers(N_TRAIN_TOPOS)), int(rng.integers(1 << 30)), a.mode,
                     dict(mc.__dict__), state, a.sigma) for _ in range(a.eps_per_iter)]
            for tid, chunk in run_eps(jobs):
                data.setdefault(tid, []).append(chunk)
            total = sum(len(c[0]) for ch in data.values() for c in ch)
            while total > a.max_samples:
                t_big = max(data, key=lambda t: len(data[t]))
                total -= len(data[t_big].pop(0)[0])
            lr = a.lr * (0.5 * (1 + np.cos(np.pi * it / a.iters)) * 0.9 + 0.1)
            loss, mu, ls = fit(model, data, a.epochs, lr, lam=a.lam)
            print(f"iter {it} samples={total} loss={loss:.4f} mu={mu:.3f} fluid_loss={ls:.4f} lr={lr:.2e} "
                  f"t={time.time()-t0:.0f}s", flush=True)
            save(model, mc, a.out, extra={"args": vars(a), "iter": it})
    finally:
        if pool_cm:
            pool_cm.terminate()


if __name__ == "__main__":
    main()
