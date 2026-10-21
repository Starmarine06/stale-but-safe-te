"""Does the learned-vs-tuned tie of the TCP emulation also appear when the same nine scenarios are replayed in the simulator?
Replays each TCP scenario's exact demand schedule in the fluid simulator (degraded telemetry, capped counters like the emulator),
with the same learned model and the re-tuned heuristic (configs/tuned_heuristic_v2.yaml), and reports capped max-utilisation
(intervals 3 onward) with paired bootstrap / Wilcoxon over scenarios, next to the emulation numbers.
  python experiments/sim_replay_tcp_scenarios.py [root [id ...]]   (default runs/emu_tcp, the nine batch-1 scenarios;
  batch 2: runs/emu_tcp_b b01 ... b18). Writes experiments/final/sim_replay_<root basename>.json"""
import sys
from pathlib import Path
import numpy as np
import yaml
from scipy import stats

ARGS = sys.argv[1:]
sys.argv = ["x"]
exec(open("experiments/guard_carried_replay.py", encoding="utf-8").read().split('print("| guard | counters')[0])
from stale_but_safe.baselines.heuristics import ECMP, TunedHeuristic

ROOT = Path(ARGS[0]) if ARGS else Path("runs/emu_tcp")
IDS = ARGS[1:] or NAMES
params = yaml.safe_load(open("configs/tuned_heuristic_v2.yaml"))["degraded"]["params"]


def traj(n, pol_fn):
    d = ROOT / n; sc = S.load(d / "scenario.json")
    mp = (d / "model.txt").read_text().strip() if (d / "model.txt").exists() else f"runs/ours_dfm_final_{n}.pt"
    m, c = load(mp); topo = S.topo_from_dict(sc["topo"])
    env = TEEnv(topo, S.sim_cfg(sc, PROFILES["degraded"], carried=True), sc["seed"]); obs = env.reset(sc["seed"])
    pol = pol_fn(m, c); pol.reset(env); mus = []; done = False
    while not done:
        dem, up = env.peek_demand(), env.link_up.copy(); act = pol.act(obs, env); mus.append(min(mu(topo, dem, act, up), 1.0))
        obs, _, done, _ = env.step(act)
    return np.array(mus)[3:]


rows = {k: [] for k in ("ecmp", "tuned", "learned")}
for n in IDS:
    rows["ecmp"].append(traj(n, lambda m, c: ECMP()).mean())
    rows["tuned"].append(traj(n, lambda m, c: TunedHeuristic(**params)).mean())
    rows["learned"].append(traj(n, lambda m, c: LearnedPolicy(m, c, "gnn")).mean())
R = {k: np.array(v) for k, v in rows.items()}
print("| scenario | " + " | ".join(IDS) + " | mean |\n|---|" + "---|" * (len(IDS) + 1))
for k, v in R.items():
    print(f"| {k} | " + " | ".join(f"{x:.3f}" for x in v) + f" | {v.mean():.3f} |")


def pair(a, b, label):
    d = R[a] - R[b]; bs = [np.mean(np.random.default_rng(s).choice(d, len(d))) for s in range(4000)]
    print(f"{label}: {a} - {b} mean {d.mean():+.4f} [{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}] lower in {(d < 0).sum()}/{len(d)} Wilcoxon p={stats.wilcoxon(d).pvalue:.3f}")


pair("learned", "tuned", "sim replay"); pair("learned", "ecmp", "sim replay"); pair("tuned", "ecmp", "sim replay")

import json
json.dump({"ids": IDS, **{k: v.tolist() for k, v in R.items()}}, open(f"experiments/final/sim_replay_{ROOT.name}.json", "w"))
