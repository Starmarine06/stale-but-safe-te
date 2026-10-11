"""PPO fine-tuning of the GNN policy on the true environment reward.

Action = logits + sigma*eps (Gaussian exploration in logit space); the executed split is
softmax(masked action-logits). Each demand's path-distribution is treated as its own action
(factored PPO: per-demand ratios, shared advantage), which keeps ratios well-behaved with
hundreds of action dimensions. Reward is the per-step improvement over the ECMP counterfactual,
matching the evaluation score:
    r_t = -[(mu - mu_ecmp)/max(mu_ecmp, .1) + 5*(loss - loss_ecmp)] - lam*churn
Telemetry is randomised exactly as in distillation (`--mode random`).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import numpy as np
import torch

from ..env.te_env import TEEnv
from ..models.features import Featurizer
from ..models.learned import ModelCfg, build, graph_static, load, save
from .scenarios import N_TRAIN_TOPOS, sample_cfg, train_topology

torch.set_num_threads(1)
LAM = 0.05


def logp_demand(a_logits, mu, log_std, valid):
    """(B,D) log-prob of the sampled logits under N(mu, sigma) summed over valid paths."""
    std = log_std.exp()
    lp = -0.5 * ((a_logits - mu) / std) ** 2 - log_std - 0.9189385
    return (lp * valid).sum(-1)


def episode(args):
    topo_id, seed, mode, mcfg, state = args
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    topo = train_topology(topo_id)
    cfg = sample_cfg(rng, mode)
    env = TEEnv(topo, cfg, seed)
    obs = env.reset(seed)
    mc = ModelCfg(**mcfg)
    model = build(mc)
    model.load_state_dict(state)
    model.eval()
    feat = Featurizer(topo, mc.H, mc.use_age, graph_static(topo))
    gen = torch.Generator().manual_seed(int(seed))
    XL, XP, V, A, LP, R, VAL, W = [], [], [], [], [], [], [], []
    done = False
    while not done:
        xl, xp, valid = feat(obs)
        with torch.no_grad():
            t = lambda a: torch.from_numpy(a)[None]
            mu, val = model(t(xl), t(xp), t(valid), feat.G)
            std = model.log_std.exp()
            a = mu + std * torch.randn(mu.shape, generator=gen)
            a = torch.where(t(valid), a, torch.zeros_like(a))          # invalid entries unused
            lp = logp_demand(a, torch.where(t(valid), mu, torch.zeros_like(mu)), model.log_std, t(valid))
            split = torch.softmax(a[0].masked_fill(~t(valid)[0], -1e9), -1).numpy()
        dem = env.peek_demand()
        feat.note_action(split)
        obs, _, done, info = env.step(split)
        scale = max(info["ecmp_max_util"], 0.1)
        r = -((info["max_util"] - info["ecmp_max_util"]) / scale + 5 * (info["loss"] - info["ecmp_loss"])) \
            - LAM * info["churn"]
        XL.append(xl); XP.append(xp); V.append(valid); A.append(a[0].numpy()); LP.append(lp[0].numpy())
        R.append(r); VAL.append(float(val)); W.append(np.clip(0.2 + 0.8 * dem / max(dem.mean(), 1e-9), 0, 5))
    f = lambda x: np.stack(x).astype(np.float32)
    return topo_id, dict(xl=f(XL), xp=f(XP), v=np.stack(V), a=f(A), lp=f(LP),
                         r=np.array(R, np.float32), val=np.array(VAL, np.float32), w=f(W))


def gae(r, v, gamma, lam):
    adv = np.zeros_like(r)
    last = 0.0
    for t in reversed(range(len(r))):
        nv = v[t + 1] if t + 1 < len(r) else 0.0
        d = r[t] + gamma * nv - v[t]
        last = d + gamma * lam * last
        adv[t] = last
    return adv, adv + v


def update(model, opt, batches, epochs, clip=0.2, vf=0.5, bs=256, target=0.25):
    model.train()
    stats = []
    for _ in range(epochs):
        for tid, d in batches:
            G = graph_static(train_topology(tid))
            n = len(d["xl"])
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                j = perm[i:i + bs]
                v = d["v"][j]
                mu, val = model(d["xl"][j], d["xp"][j], v, G)
                lp = logp_demand(d["a"][j], torch.where(v, mu, torch.zeros_like(mu)), model.log_std, v)
                ratio = torch.exp(lp - d["lp"][j])
                adv = d["adv"][j][:, None]
                w = d["w"][j]
                s1, s2 = ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv
                pg = -(w * torch.min(s1, s2)).sum() / w.sum()
                vl = ((val - d["ret"][j]) ** 2).mean()
                loss = pg + vf * vl
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                with torch.no_grad():
                    model.log_std.clamp_(-2.5, -0.3)
                dev = float((ratio - 1).abs().mean().item())
                stats.append((pg.item(), vl.item(), dev))
                if dev > target:          # KL-style early stop
                    model.eval()
                    return np.mean(stats, 0)
    model.eval()
    return np.mean(stats, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--mode", default="random", choices=["random", "ideal"])
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--eps-per-iter", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model, mc = load(a.init)
    rng = np.random.default_rng(a.seed + 99)
    torch.manual_seed(a.seed)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for it in range(a.iters):
            state = {k: v.clone() for k, v in model.state_dict().items()}
            jobs = [(int(rng.integers(N_TRAIN_TOPOS)), int(rng.integers(1 << 30)), a.mode, dict(mc.__dict__), state)
                    for _ in range(a.eps_per_iter)]
            per_topo: dict = {}
            rets = []
            for tid, ep in pool.imap_unordered(episode, jobs):
                adv, ret = gae(ep["r"], ep["val"], a.gamma, 0.9)
                ep["adv"], ep["ret"] = adv, ret
                rets.append(float(ep["r"].mean()))
                per_topo.setdefault(tid, []).append(ep)
            allv = np.concatenate([e["adv"] for eps in per_topo.values() for e in eps])
            mu_a, sd_a = allv.mean(), allv.std() + 1e-8
            batches = []
            for tid, eps in per_topo.items():
                cat = {k: torch.from_numpy(np.concatenate([e[k] for e in eps])) for k in
                       ("xl", "xp", "v", "a", "lp", "w", "adv", "ret")}
                cat["adv"] = (cat["adv"] - mu_a) / sd_a
                batches.append((tid, cat))
            st = update(model, opt, batches, a.epochs)
            print(f"iter {it} mean_r={np.mean(rets):.4f} pg={st[0]:.4f} vl={st[1]:.4f} |ratio-1|={st[2]:.4f} "
                  f"sigma={model.log_std.exp().item():.3f} t={time.time()-t0:.0f}s", flush=True)
            if (it + 1) % 5 == 0 or it == a.iters - 1:
                save(model, mc, a.out, extra={"args": vars(a), "iter": it})


if __name__ == "__main__":
    main()
