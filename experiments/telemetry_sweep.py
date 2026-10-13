"""One impairment at a time: which telemetry parameter drives the ranking? (10 Zoo graphs n<=25 + Abilene + GEANT)
The tuned heuristic uses the parameters tuned for the nearest standard profile (no per-point re-tuning, stated limitation)."""
import numpy as np
import yaml

from stale_but_safe.audit import _tuned_profile
from stale_but_safe.baselines.heuristics import TunedHeuristic
from stale_but_safe.env import real
from stale_but_safe.env.te_env import EnvCfg
from stale_but_safe.env.telemetry import TelemetryCfg
from stale_but_safe.env.zoo import load_split, zoo_topology
from stale_but_safe.eval.compare import episode_stats
from stale_but_safe.models.learned import LearnedPolicy, load

tuned = yaml.safe_load(open("configs/tuned_heuristic.yaml"))
m, c = load("runs/ours_dfm_final.pt")
zoo = [t for t in (zoo_topology(n) for n in load_split("test_subset")) if t.n <= 25][:10]
ab, atr = real.load("abilene"); ge, gtr = real.load("geant")
cases = [("Zoo (10 graphs)", zoo, None), ("Abilene real", [ab], atr), ("GEANT real", [ge], gtr)]
base = dict(P=1, d=0, sigma=0.0, p=0.0)
grid = [("P (polling period)", "P", [1, 2, 5, 10, 20]), ("d (delay)", "d", [0, 1, 3, 5, 10]),
        ("sigma (noise)", "sigma", [0.0, 0.1, 0.2, 0.4]), ("p (missing counters)", "p", [0.0, 0.1, 0.2, 0.4])]
print("| case | parameter | value | tuned gain % | learned gain % | tuned p99 regret | learned p99 regret |")
print("|---|---|---|---|---|---|---|")
for cname, topos, trace in cases:
    for label, key, vals in grid:
        for v in vals:
            t = TelemetryCfg(**{**base, key: v})
            pols = {"tuned": TunedHeuristic(**tuned[_tuned_profile(t)]["params"]), "learned": LearnedPolicy(m, c, "g")}
            out = {}
            for name, pol in pols.items():
                g, reg = [], []
                for tp in topos:
                    cfg = EnvCfg(load=0.8, horizon=100, telemetry=t, trace=trace)
                    for s in range(3 if trace is None else 6):
                        r = episode_stats(tp, cfg, pol, 12000 + s)
                        g.append(1 - r["mu"] / r["ecmp_mu"]); reg.append(r["regret"])
                out[name] = (100 * np.mean(g), np.quantile(np.concatenate(reg), .99))
            print(f"| {cname} | {label} | {v} | {out['tuned'][0]:.1f} | {out['learned'][0]:.1f} | {out['tuned'][1]:+.3f} | {out['learned'][1]:+.3f} |", flush=True)
