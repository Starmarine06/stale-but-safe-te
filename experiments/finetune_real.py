"""Pre-train on synthetic graphs (zero-shot GNN) then FINE-TUNE on a network's own early traffic history.

  python experiments/finetune_real.py abilene --init runs/ours_dfm_final.pt --out runs/ft_abilene.pt

Fine-tuning data: rollouts on the first 70% of the real trace (random telemetry, 3 loads, student with logit noise),
same differentiable fluid loss as pre-training. Evaluation: last 30% of the trace (windows never seen), compared with
the zero-shot model, the DOTE-style per-network MLP, the tuned heuristic and ECMP.
"""
import argparse
import json
import time

import numpy as np
import torch

from stale_but_safe.baselines.heuristics import ECMP, TunedHeuristic
from stale_but_safe.env import real
from stale_but_safe.env.te_env import EnvCfg, TEEnv
from stale_but_safe.env.telemetry import PROFILES
from stale_but_safe.env.traffic import ecmp_splits
from stale_but_safe.eval.compare import ecmp_score, episode_stats, score
from stale_but_safe.models.features import Featurizer
from stale_but_safe.models.learned import LearnedPolicy, graph_static, load, save
from stale_but_safe.train.dfm import cap_t, fluid_loss
from stale_but_safe.train.scenarios import sample_telemetry
import yaml

torch.set_num_threads(2)
ap = argparse.ArgumentParser()
ap.add_argument("network")
ap.add_argument("--init", default="runs/ours_dfm_final.pt")
ap.add_argument("--out", required=True)
ap.add_argument("--iters", type=int, default=6)
ap.add_argument("--eps", type=int, default=24)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--sigma", type=float, default=0.3)
ap.add_argument("--eval-seeds", type=int, default=10)
ap.add_argument("--result", required=True)
a = ap.parse_args()

topo, trace = real.load(a.network)
cut = int(0.7 * len(trace))
tr, te = trace[:cut], trace[cut:]
model, mc = load(a.init)
G, cap = graph_static(topo), cap_t(topo)
opt = torch.optim.Adam(model.parameters(), lr=a.lr)
rng = np.random.default_rng(0)


def rollout(seed):
    cfg = EnvCfg(load=float(rng.choice([0.6, 0.8, 1.0])), horizon=60, telemetry=sample_telemetry(rng, "random"), trace=tr)
    env = TEEnv(topo, cfg, seed)
    obs = env.reset(seed)
    feat = Featurizer(topo, mc.H, mc.use_age, G, mc.use_model)
    gen = torch.Generator().manual_seed(seed)
    S = {k: [] for k in ("xl", "xp", "v", "dm", "pv", "em")}
    done = False
    while not done:
        xl, xp, valid = feat(obs)
        S["xl"].append(xl); S["xp"].append(xp); S["v"].append(valid)
        S["dm"].append(env.peek_demand().astype(np.float32)); S["pv"].append(feat.prev.astype(np.float32))
        t = lambda x: torch.from_numpy(x)[None]
        with torch.no_grad():
            mu, _ = model(t(xl), t(xp), t(valid), G)
            act = torch.softmax((mu + a.sigma * torch.randn(mu.shape, generator=gen))[0].masked_fill(~t(valid)[0], -1e9), -1).numpy()
        feat.note_action(act)
        obs, _, done, info = env.step(act)
        S["em"].append(info["ecmp_max_util"])
    return {k: torch.from_numpy(np.stack(v)).float() if k != "v" else torch.from_numpy(np.stack(v)) for k, v in S.items()}


t0 = time.time()
for it in range(a.iters):
    eps = [rollout(1000 * (it + 1) + e) for e in range(a.eps)]
    D = {k: torch.cat([e[k] for e in eps]) for k in eps[0]}
    model.train()
    n, perm, tot = len(D["xl"]), None, []
    for _ in range(2):
        perm = torch.randperm(n)
        for i in range(0, n, 32):
            j = perm[i:i + 32]
            lg, _ = model(D["xl"][j], D["xp"][j], D["v"][j], G)
            loss, mu, ls = fluid_loss(torch.softmax(lg, -1), D["dm"][j], G, cap, D["em"][j], D["pv"][j])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot.append(loss.item())
    model.eval()
    print(f"iter {it} loss {np.mean(tot):.4f} t={time.time() - t0:.0f}s", flush=True)
save(model, mc, a.out)

tuned = yaml.safe_load(open("configs/tuned_heuristic.yaml"))
zs, zc = load(a.init)
out = {"results": []}
seeds = [9000 + i for i in range(a.eval_seeds)]
for ld in (0.6, 0.8, 1.0):
    for pn in ("ideal", "snmp", "degraded"):
        cfg = EnvCfg(load=ld, horizon=100, telemetry=PROFILES[pn], trace=te)
        meth = {"ecmp": ECMP(), "tuned": TunedHeuristic(**tuned[pn]["params"]),
                "gnn-zeroshot": LearnedPolicy(zs, zc, "zs"), "gnn-finetuned": LearnedPolicy(model, mc, "ft")}
        for mname, pol in meth.items():
            rows = [episode_stats(topo, cfg, pol, s) for s in seeds]
            reg = np.concatenate([r["regret"] for r in rows])
            out["results"].append({"topo": f"{a.network}-test@{ld}", "n": topo.n, "profile": pn, "method": mname,
                                   "gain": float(np.mean([1 - r["mu"] / r["ecmp_mu"] for r in rows])),
                                   "delta": float(np.mean([score(r) - ecmp_score(r) for r in rows])),
                                   "regret_p99": float(np.quantile(reg, 0.99))})
        print(a.network, ld, pn, " ".join(f"{r['method']}={100 * r['gain']:.1f}" for r in out["results"][-4:]), flush=True)
        json.dump(out, open(a.result, "w"))
