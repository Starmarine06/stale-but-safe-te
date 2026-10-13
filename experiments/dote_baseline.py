"""DOTE-style per-network baseline vs our zero-shot GNN on REAL traces (temporal train/test split).

  python experiments/dote_baseline.py abilene --out experiments/final/dote_abilene.json

DOTE (Perry et al., NSDI'23) trains a DNN on a network's own traffic history to output split ratios that
minimise max link utilisation directly. Here: an MLP maps the last H=3 *impaired* demand estimates to split
logits, trained through the same differentiable fluid loss as our policy, on the first 70% of the real trace of
that network; everything is evaluated on windows of the last 30% (never seen). Two variants: trained with ideal
telemetry (original DOTE assumption) and with degraded telemetry (matched). Our GNN was trained on synthetic
graphs only (zero-shot), so this baseline has an in-distribution advantage.
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

from stale_but_safe.baselines.heuristics import ECMP, Policy, TunedHeuristic
from stale_but_safe.env import real
from stale_but_safe.env.te_env import EnvCfg, TEEnv
from stale_but_safe.env.telemetry import PROFILES
from stale_but_safe.env.traffic import ecmp_splits
from stale_but_safe.eval.compare import ecmp_score, episode_stats, score
from stale_but_safe.models.learned import LearnedPolicy, graph_static, load
from stale_but_safe.train.dfm import cap_t, fluid_loss

torch.set_num_threads(2)
H = 3


class DoteMLP(nn.Module):
    def __init__(self, D, k, hid=256):
        super().__init__()
        self.D, self.k = D, k
        self.net = nn.Sequential(nn.Linear(D * H, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, D * k))

    def forward(self, x, valid):
        lg = self.net(x).view(-1, self.D, self.k).masked_fill(~valid[None], -1e9)
        return torch.softmax(lg, -1)


class DotePolicy(Policy):
    def __init__(self, model, topo, name="dote"):
        self.m, self.name, self.unit = model, name, float(topo.cap.mean())
        self.valid = torch.from_numpy(topo.path_mask)

    def reset(self, env):
        super().reset(env)
        self.hist = []

    def act(self, obs, env):
        self.hist.append(np.asarray(obs["demand"], np.float32) / self.unit)
        h = (self.hist[-H:] + [self.hist[0]] * H)[:H] if len(self.hist) < H else self.hist[-H:]
        x = torch.from_numpy(np.concatenate(h))[None]
        with torch.no_grad():
            sp = self.m(x, self.valid)[0].numpy().astype(float)
        valid = self.topo.valid_paths(obs["link_up"])
        sp = np.where(valid, sp, 0.0)
        return sp / np.maximum(sp.sum(1, keepdims=True), 1e-9)


def collect(topo, trace_tr, prof, loads, n_ep, horizon=60):
    X, DM, EM = [], [], []
    unit = float(topo.cap.mean())
    for li, ld in enumerate(loads):
        for e in range(n_ep):
            cfg = EnvCfg(load=ld, horizon=horizon, telemetry=PROFILES[prof], trace=trace_tr)
            env = TEEnv(topo, cfg, 100 + 1000 * li + e)
            obs = env.reset(100 + 1000 * li + e)
            pol, hist, done = ECMP(), [], False
            pol.reset(env)
            while not done:
                hist.append(np.asarray(obs["demand"], np.float32) / unit)
                h = (hist[-H:] + [hist[0]] * H)[:H] if len(hist) < H else hist[-H:]
                dem = env.peek_demand().astype(np.float32)
                obs, _, done, info = env.step(pol.act(obs, env))
                X.append(np.concatenate(h)); DM.append(dem); EM.append(info["ecmp_max_util"])
    return torch.tensor(np.stack(X)), torch.tensor(np.stack(DM)), torch.tensor(np.array(EM, np.float32))


def train_dote(topo, trace_tr, prof, loads, n_ep, epochs, seed=0):
    torch.manual_seed(seed)
    X, DM, EM = collect(topo, trace_tr, prof, loads, n_ep)
    G, cap = graph_static(topo), cap_t(topo)
    valid = torch.from_numpy(topo.path_mask)
    m = DoteMLP(topo.D, topo.k)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    ecmp = torch.tensor(ecmp_splits(topo), dtype=torch.float32)
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 64):
            j = perm[i:i + 64]
            sp = m(X[j], valid)
            loss, mu, ls = fluid_loss(sp, DM[j], G, cap, EM[j], ecmp[None].expand(len(j), -1, -1), lam=0.0)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            tot += loss.item() * len(j)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  dote[{prof}] epoch {ep} loss {tot / n:.4f}", flush=True)
    m.eval()
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("network")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--loads", default="0.6,0.8,1.0")
    ap.add_argument("--episodes", type=int, default=60, help="training episodes per load")
    ap.add_argument("--epochs", type=int, default=25)
    a = ap.parse_args()

    tuned = yaml.safe_load(open("configs/tuned_heuristic.yaml"))
    topo, trace = real.load(a.network)
    cut = int(0.7 * len(trace))
    tr, te = trace[:cut], trace[cut:]
    loads = [float(x) for x in a.loads.split(",")]
    print(f"{a.network}: train {len(tr)} intervals, test {len(te)} intervals", flush=True)
    dote = {p: train_dote(topo, tr, p, loads, a.episodes, a.epochs) for p in ("ideal", "degraded")}
    gnn = {f"gnn-ours-s{s}": load(p) for s, p in enumerate(
        ["runs/ours_dfm_final.pt", "runs/ours_dfm_final_s1.pt", "runs/ours_dfm_final_s2.pt", "runs/ours_dfm_final_s3.pt"])}
    out = {"args": vars(a), "results": []}
    seeds = [9000 + i for i in range(a.seeds)]
    t0 = time.time()
    for ld in loads:
        for pn in ("ideal", "snmp", "degraded"):
            cfg = EnvCfg(load=ld, horizon=100, telemetry=PROFILES[pn], trace=te)
            meth = {"ecmp": ECMP(), "tuned": TunedHeuristic(**tuned[pn]["params"]),
                    "dote-ideal": DotePolicy(dote["ideal"], topo, "dote-ideal"),
                    "dote-degraded": DotePolicy(dote["degraded"], topo, "dote-degraded")}
            for k, (m, c) in gnn.items():
                meth[k] = LearnedPolicy(m, c, k)
            for mname, pol in meth.items():
                rows = [episode_stats(topo, cfg, pol, s) for s in seeds]
                reg = np.concatenate([r["regret"] for r in rows])
                out["results"].append({
                    "topo": f"{a.network}-test@{ld}", "n": topo.n, "profile": pn, "method": mname,
                    "delta": float(np.mean([score(r) - ecmp_score(r) for r in rows])),
                    "gain": float(np.mean([1 - r["mu"] / r["ecmp_mu"] for r in rows])),
                    "loss": float(np.mean([r["loss"] for r in rows])),
                    "regret_p99": float(np.quantile(reg, 0.99)), "regret_mean": float(reg.mean())})
            print(f"{a.network} load={ld} {pn:8s} done t={time.time() - t0:.0f}s", flush=True)
            json.dump(out, open(a.out, "w"))
    json.dump(out, open(a.out, "w"))
