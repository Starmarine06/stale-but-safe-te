"""Do the conclusions survive impairments that look more like measured SNMP collection than the stylised profiles?
Variants of the degraded profile (P=5, d=3, sigma=0.2, p=0.1): gross errors on a fraction of readings (heavy-tailed errors as in a
case study of SNMP collection), whole-round loss, poll-timing jitter, and all three together. The learned models were trained without
these impairments. The heuristic is evaluated with the parameters tuned for the plain degraded profile and, for the harsher variants,
re-tuned for the variant (80 trials, tuner seeds 100 and 101, four synthetic graphs). 10 Zoo graphs (n <= 25), 3 seeds, load 0.8.
  python experiments/telemetry_realism.py > experiments/final/table_telemetry_realism.md"""
import glob

import numpy as np
import yaml

from stale_but_safe.baselines.heuristics import TunedHeuristic, tune_heuristic
from stale_but_safe.env.te_env import EnvCfg
from stale_but_safe.env.telemetry import TelemetryCfg
from stale_but_safe.env.topology import synthetic
from stale_but_safe.env.zoo import load_split, zoo_topology
from stale_but_safe.eval.compare import episode_stats
from stale_but_safe.eval.tune import TUNE_TOPOS, score as tscore
from stale_but_safe.models.learned import LearnedPolicy, load

DEG = dict(P=5, d=3, sigma=0.2, p=0.1)
VARIANTS = {"degraded (plain)": {}, "+ 2% gross errors": dict(outlier=0.02), "+ 5% gross errors": dict(outlier=0.05),
            "+ 10% lost poll rounds": dict(round_drop=0.1), "+ poll jitter 0.3": dict(jitter=0.3),
            "all three (2%, 5% rounds, jitter 0.3)": dict(outlier=0.02, round_drop=0.05, jitter=0.3)}
RETUNE = ("+ 5% gross errors", "all three (2%, 5% rounds, jitter 0.3)")
v2 = yaml.safe_load(open("configs/tuned_heuristic_v2.yaml"))["degraded"]["params"]
models = [load(p) for p in ["runs/ours_dfm_final.pt"] + sorted(glob.glob("runs/ours_dfm_final_s[123].pt"))]
zoo = [t for t in (zoo_topology(n) for n in load_split("test_subset")) if t.n <= 25][:10]
tt = [synthetic(a, n, s) for a, n, s in TUNE_TOPOS]
rng = np.random.default_rng(0)


def evaluate(pol, tele):
    g, reg = [], []
    for tp in zoo:
        gg = []
        for s in range(3):
            r = episode_stats(tp, EnvCfg(load=0.8, horizon=100, telemetry=tele), pol, 12000 + s)
            gg.append(1 - r["mu"] / r["ecmp_mu"]); reg.append(r["regret"])
        g.append(np.mean(gg))
    return np.array(g), float(np.quantile(np.concatenate(reg), .99))


def ci(x):
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(4000)]
    return np.percentile(b, [2.5, 97.5])


print(f"| impairment | heuristic parameters | heuristic gain % | learned gain % (mean of {len(models)} models) | learned - heuristic [95% CI over graphs], points | p99 regret heuristic / learned |\n|---|---|---|---|---|---|")
for name, extra in VARIANTS.items():
    tele = TelemetryCfg(**DEG, **extra)
    lg, lp = zip(*[evaluate(LearnedPolicy(m, c, "g"), tele) for m, c in models])
    lg, lp = np.mean(lg, axis=0), float(np.mean(lp))
    arms = {"tuned for plain degraded": [v2]}
    if name in RETUNE:
        arms["re-tuned for this variant (2 seeds)"] = []
        for k in range(2):
            best, _ = tune_heuristic(lambda: EnvCfg(load=0.8, horizon=60, telemetry=tele), lambda pol, m: tscore(pol, m, tt), budget=80, seed=100 + k)
            arms["re-tuned for this variant (2 seeds)"].append({a: float(b) for a, b in best.items()})
    for arm, plist in arms.items():
        hs = [evaluate(TunedHeuristic(**p), tele) for p in plist]
        hg, hp = np.mean([h[0] for h in hs], axis=0), float(np.mean([h[1] for h in hs]))
        d = 100 * (lg - hg); lo, hi = ci(d)
        print(f"| {name} | {arm} | {100 * hg.mean():.1f} | {100 * lg.mean():.1f} | {d.mean():+.1f} [{lo:+.1f}, {hi:+.1f}] | {hp:+.3f} / {lp:+.3f} |", flush=True)
