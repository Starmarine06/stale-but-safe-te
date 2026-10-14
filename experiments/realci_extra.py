"""Germany50 and Brain with the re-tuned heuristic: learned (4 seeds) vs heuristic (configs/tuned_heuristic_v2.yaml), load 0.8, 4 profiles, W windows (seeds 20000+).
  python experiments/realci_extra.py germany50 [W] > experiments/final/table_realci_extra_germany50.md   (raw: experiments/final/realci_extra_<net>.json)"""
import json
import sys

import numpy as np
import yaml

from stale_but_safe.baselines.heuristics import TunedHeuristic
from stale_but_safe.env import real
from stale_but_safe.env.te_env import EnvCfg
from stale_but_safe.env.telemetry import PROFILES
from stale_but_safe.eval.compare import episode_stats
from stale_but_safe.models.learned import LearnedPolicy, load

net = sys.argv[1]
W = int(sys.argv[2]) if len(sys.argv) > 2 else 12
v2 = yaml.safe_load(open("configs/tuned_heuristic_v2.yaml"))
models = [load(p) for p in ["runs/ours_dfm_final.pt", "runs/ours_dfm_final_s1.pt", "runs/ours_dfm_final_s2.pt", "runs/ours_dfm_final_s3.pt"]]
topo, trace = real.load(net)
rng = np.random.default_rng(0)
out = {}
print(f"{net}, load 0.8, {W} windows per profile; learned = mean of 4 training seeds; difference = learned minus re-tuned heuristic in points with a 95% bootstrap CI over windows.\n")
print("| telemetry | heuristic gain % | learned gain % | difference [95% CI] | verdict | windows |\n|---|---|---|---|---|---|")
for pn in ("ideal", "rich", "snmp", "degraded"):
    cfg = EnvCfg(load=0.8, horizon=100, telemetry=PROFILES[pn], trace=trace)
    H, L = [], []
    for s in range(W):
        seed = 20000 + s
        r = episode_stats(topo, cfg, TunedHeuristic(**v2[pn]["params"]), seed)
        if r["ecmp_mu"] <= 1e-9:
            continue
        H.append(1 - r["mu"] / r["ecmp_mu"])
        L.append(np.mean([1 - episode_stats(topo, cfg, LearnedPolicy(m, c, "g"), seed)["mu"] / r["ecmp_mu"] for m, c in models]))
    out[pn] = {"heur": H, "learned": L}
    json.dump(out, open(f"experiments/final/realci_extra_{net}.json", "w"))
    h, l = 100 * np.array(H), 100 * np.array(L)
    df = l - h
    bs = [df[rng.integers(0, len(df), len(df))].mean() for _ in range(4000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    v = "L" if lo > 0 else "H" if hi < 0 else "="
    print(f"| {pn} | {h.mean():.1f} | {l.mean():.1f} | {df.mean():+.1f} [{lo:+.1f}, {hi:+.1f}] | {v} | {len(df)} |", flush=True)
