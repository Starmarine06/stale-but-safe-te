"""Rule-install latency L (intervals) one at a time, other telemetry ideal: tuned vs learned (10 Zoo graphs n<=25, Abilene, GEANT)."""
import numpy as np, yaml
from stale_but_safe.baselines.heuristics import TunedHeuristic
from stale_but_safe.env import real
from stale_but_safe.env.te_env import EnvCfg
from stale_but_safe.env.telemetry import PROFILES
from stale_but_safe.env.zoo import load_split, zoo_topology
from stale_but_safe.eval.compare import episode_stats
from stale_but_safe.models.learned import LearnedPolicy, load
tuned = yaml.safe_load(open("configs/tuned_heuristic.yaml")); m, c = load("runs/ours_dfm_final.pt")
zoo = [t for t in (zoo_topology(n) for n in load_split("test_subset")) if t.n <= 25][:10]
ab, atr = real.load("abilene"); ge, gtr = real.load("geant")
print("| case | install latency L | tuned gain % | learned gain % | tuned p99 regret | learned p99 regret |")
print("|---|---|---|---|---|---|")
for cname, topos, trace in [("Zoo (10 graphs)", zoo, None), ("Abilene real", [ab], atr), ("GEANT real", [ge], gtr)]:
    for L in (0, 1, 2, 4, 8):
        out = {}
        for name, pol in (("tuned", TunedHeuristic(**tuned["ideal"]["params"])), ("learned", LearnedPolicy(m, c, "g"))):
            g, reg = [], []
            for tp in topos:
                cfg = EnvCfg(load=0.8, horizon=100, telemetry=PROFILES["ideal"], L=L, trace=trace)
                for s in range(3 if trace is None else 6):
                    r = episode_stats(tp, cfg, pol, 13000 + s); g.append(1 - r["mu"] / r["ecmp_mu"]); reg.append(r["regret"])
            out[name] = (100 * np.mean(g), np.quantile(np.concatenate(reg), .99))
        print(f"| {cname} | {L} | {out['tuned'][0]:.1f} | {out['learned'][0]:.1f} | {out['tuned'][1]:+.3f} | {out['learned'][1]:+.3f} |", flush=True)
