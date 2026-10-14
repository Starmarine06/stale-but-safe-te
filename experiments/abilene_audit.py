"""Audit of the Abilene discrepancy: same windows, different heuristic parameters / window samples (load 0.8).
Window sets: A = seeds 20000+ (profile-level tables), B = seeds 15000+ (delay-grid CI). Heuristics: yaml = configs/tuned_heuristic.yaml
(40 trials, 1 seed, per profile); grid = best-of-5 80-trial parameters for cell (d=3,P=5) from delay_ci.json.
  python experiments/abilene_audit.py > experiments/final/table_abilene_audit.md"""
import json
import numpy as np
import yaml
from stale_but_safe.baselines.heuristics import TunedHeuristic
from stale_but_safe.env import real
from stale_but_safe.env.te_env import EnvCfg
from stale_but_safe.env.telemetry import PROFILES, TelemetryCfg
from stale_but_safe.eval.compare import episode_stats
from stale_but_safe.models.learned import LearnedPolicy, load

yml = yaml.safe_load(open("configs/tuned_heuristic.yaml"))
grid = {(r["d"], r["P"]): r["params"] for r in json.load(open("experiments/final/delay_ci.json"))}
models = [load(p) for p in ["runs/ours_dfm_final.pt", "runs/ours_dfm_final_s1.pt", "runs/ours_dfm_final_s2.pt", "runs/ours_dfm_final_s3.pt"]]
topo, trace = real.load("abilene")
sets = {"A (20000+, 12 windows)": range(20000, 20012), "B (15000+, 16 windows)": range(15000, 15016)}
teles = {"snmp": PROFILES["snmp"], "degraded": PROFILES["degraded"], "clean d=3,P=5": TelemetryCfg(5, 3, 0.0, 0.0)}
rng = np.random.default_rng(0)
print("| telemetry | window set | heuristic params | heuristic % | learned % | learned - heuristic [95% CI over windows] |\n|---|---|---|---|---|---|")
for tn, tele in teles.items():
    hs = {}
    if tn in yml: hs["yaml (per-profile, 40 trials)"] = yml[tn]["params"]
    hs["grid best-of-5 (d=3,P=5)"] = grid[(3, 5)]
    for sn, seeds in sets.items():
        cfg = EnvCfg(load=0.8, horizon=100, telemetry=tele, trace=trace)
        ecm = {s: episode_stats(topo, cfg, TunedHeuristic(**grid[(3, 5)]), s)["ecmp_mu"] for s in seeds}
        L = {s: np.mean([1 - episode_stats(topo, cfg, LearnedPolicy(m, c, "g"), s)["mu"] / ecm[s] for m, c in models]) for s in seeds if ecm[s] > 1e-9}
        for hn, hp in hs.items():
            H = {s: 1 - episode_stats(topo, cfg, TunedHeuristic(**hp), s)["mu"] / ecm[s] for s in L}
            df = 100 * np.array([L[s] - H[s] for s in L])
            bs = [df[rng.integers(0, len(df), len(df))].mean() for _ in range(3000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            print(f"| {tn} | {sn} | {hn} | {100*np.mean(list(H.values())):.1f} | {100*np.mean(list(L.values())):.1f} | {df.mean():+.1f} [{lo:+.1f}, {hi:+.1f}] |", flush=True)
