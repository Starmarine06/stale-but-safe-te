"""Sim-vs-emulation agreement for the runs in runs/emu (same scenario, same policies/profiles)."""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from stale_but_safe.baselines.heuristics import ECMP, TunedHeuristic
from stale_but_safe.emu import analyze as A
from stale_but_safe.emu import scenario as S
from stale_but_safe.guard.guard import Guard, GuardCfg
from stale_but_safe.models.learned import LearnedPolicy, load

run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/emu")
sc = S.load(run_dir / "scenario.json")
tuned = yaml.safe_load(open("configs/tuned_heuristic.yaml"))


def make_policy(spec, profile):
    eps = None
    if spec.startswith("guard"):
        head, spec = spec.split("@", 1)
        eps = float(head[5:] or 0.05)
    if spec == "ecmp":
        pol = ECMP()
    elif spec.startswith("tuned:"):
        pol = TunedHeuristic(**tuned[spec.split(":", 1)[1]]["params"])
    else:
        m, c = load(spec.split(":", 1)[1])
        pol = LearnedPolicy(m, c, "gnn")
    return Guard(pol, GuardCfg(eps=eps)) if eps is not None else pol


runs = [("ecmp", "ideal"), ("tuned:snmp", "snmp"), ("tuned:degraded", "degraded"),
        ("model:runs/ours_dfm_final.pt", "snmp"), ("model:runs/ours_dfm_final.pt", "degraded"),
        ("model:runs/ideal_dfm_final.pt", "degraded"), ("guard0.05@model:runs/ours_dfm_final.pt", "degraded")]
res = A.compare(run_dir, sc, runs, make_policy)
print("| policy | telemetry | emu max-util | sim max-util | Pearson | Spearman | emu loss | sim loss | ctrl latency ms | flow-mods/min | fallback |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for r in res:
    g = r["guard"] or {}
    fb = f"{g.get('fallback', 0)}/{g.get('steps', 0)}" if g else "-"
    print(f"| {r['spec'].replace('model:runs/', '').replace('.pt', '')} | {r['profile']} | {r['emu_mu']:.3f} | {r['sim_mu']:.3f} | {r['pearson']:.2f} | {r['spearman']:.2f} "
          f"| {r['emu_loss']:.4f} | {r['sim_loss']:.4f} | {r['latency_ms']:.0f} | {r['flowmods_per_min']:.1f} | {fb} |")
mus_e = np.array([r["emu_mu"] for r in res]); mus_s = np.array([r["sim_mu"] for r in res])
from scipy import stats
print(f"\nacross {len(res)} runs: Pearson(emu, sim mean max-util) = {stats.pearsonr(mus_e, mus_s)[0]:.2f}; "
      f"Kendall tau of method ranking = {stats.kendalltau(mus_e, mus_s)[0]:.2f}; mean |emu-sim| = {np.abs(mus_e - mus_s).mean():.3f}")
json.dump(res, open("experiments/final/emu_compare.json", "w"), indent=1, default=float)
