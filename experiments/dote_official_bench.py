"""Benchmark against DOTE with the released architecture and loss (src/stale_but_safe/baselines/dote_official.py).

  python experiments/dote_official_bench.py abilene --out experiments/final/dote_official_abilene.json

Same protocol as dote_baseline.py (per-network training on the first 70% of the real trace, evaluation on windows of
the last 30%, same seeds, loads and telemetry profiles) so the rows are directly comparable with the DOTE-style rows.
DOTE is trained with ideal and with degraded telemetry; the zero-shot GNN and the re-tuned heuristic are run alongside.
"""
import argparse
import json
import time

import numpy as np
import torch
import yaml

from stale_but_safe.baselines.dote_official import HIST, DoteNet, DoteOfficialPolicy, dote_loss, splits_from_weights
from stale_but_safe.baselines.heuristics import ECMP, TunedHeuristic
from stale_but_safe.env import real
from stale_but_safe.env.te_env import EnvCfg, TEEnv
from stale_but_safe.env.telemetry import PROFILES
from stale_but_safe.eval.compare import ecmp_score, episode_stats, score
from stale_but_safe.models.learned import LearnedPolicy, graph_static, load
from stale_but_safe.train.dfm import cap_t

torch.set_num_threads(2)


def collect(topo, trace_tr, prof, loads, n_ep, horizon=60):
    X, DM = [], []
    unit = float(topo.cap.mean())
    for li, ld in enumerate(loads):
        for e in range(n_ep):
            seed = 100 + 1000 * li + e
            env = TEEnv(topo, EnvCfg(load=ld, horizon=horizon, telemetry=PROFILES[prof], trace=trace_tr), seed)
            obs = env.reset(seed)
            pol, hist, done = ECMP(), [], False
            pol.reset(env)
            while not done:
                hist.append(np.asarray(obs["demand"], np.float32) / unit)
                h = hist[-HIST:]
                h = [h[0]] * (HIST - len(h)) + h
                dem = env.peek_demand().astype(np.float32)
                obs, _, done, _ = env.step(pol.act(obs, env))
                X.append(np.concatenate(h)); DM.append(dem)
    return torch.tensor(np.stack(X)), torch.tensor(np.stack(DM))


def train(topo, trace_tr, prof, loads, n_ep, epochs, seed=0):
    torch.manual_seed(seed)
    X, DM = collect(topo, trace_tr, prof, loads, n_ep)
    inc, cap = torch.tensor(topo.inc, dtype=torch.float32), cap_t(topo)
    mask = torch.from_numpy(topo.path_mask.astype(np.float32))
    m = DoteNet(topo.D * HIST, topo.D * topo.k)
    opt = torch.optim.Adam(m.parameters())            # default lr, as in the release
    n = len(X)
    for ep in range(epochs):
        perm, tot = torch.randperm(n), 0.0
        for i in range(0, n, 64):
            j = perm[i:i + 64]
            loss, mu = dote_loss(splits_from_weights(m(X[j]), mask, topo.D, topo.k), DM[j], inc, cap)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += mu * len(j)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  dote-official[{prof}] epoch {ep} mean max-util {tot / n:.4f}", flush=True)
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
    dote = {p: train(topo, tr, p, loads, a.episodes, a.epochs) for p in ("ideal", "degraded")}
    gnn = {f"gnn-ours-s{s}": load(p) for s, p in enumerate(
        ["runs/ours_dfm_final.pt", "runs/ours_dfm_final_s1.pt", "runs/ours_dfm_final_s2.pt", "runs/ours_dfm_final_s3.pt"])}
    out = {"args": vars(a), "results": []}
    seeds = [9000 + i for i in range(a.seeds)]
    t0 = time.time()
    for ld in loads:
        for pn in ("ideal", "snmp", "degraded"):
            cfg = EnvCfg(load=ld, horizon=100, telemetry=PROFILES[pn], trace=te)
            meth = {"ecmp": ECMP(), "tuned": TunedHeuristic(**tuned[pn]["params"]),
                    "dote-official-ideal": DoteOfficialPolicy(dote["ideal"], topo, "dote-official-ideal"),
                    "dote-official-degraded": DoteOfficialPolicy(dote["degraded"], topo, "dote-official-degraded")}
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
