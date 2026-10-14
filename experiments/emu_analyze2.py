"""Multi-scenario sim-vs-emulation agreement and operational cost (runs/emu2/<scenario>[ _h0.2 ])."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

from stale_but_safe.baselines.heuristics import ECMP, TunedHeuristic
from stale_but_safe.emu import analyze as A
from stale_but_safe.emu import scenario as S
from stale_but_safe.guard.guard import Guard, GuardCfg
from stale_but_safe.models.learned import LearnedPolicy, load

root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/emu2")
tuned = yaml.safe_load(open("configs/tuned_heuristic.yaml"))
tuned_v2 = yaml.safe_load(open("configs/tuned_heuristic_v2.yaml"))
M = "model:runs/ours_dfm_final.pt"


def make_policy(spec, profile):
    guard = None
    if spec.startswith("guard"):
        head, spec = spec.split("@", 1)
        guard = GuardCfg(eps=float(head[7:]), layers=(True, True, False))
    if spec == "ecmp":
        pol = ECMP()
    elif spec.startswith("tunedv2:"):
        pol = TunedHeuristic(**tuned_v2[spec.split(":", 1)[1]]["params"])
    elif spec.startswith("tuned:"):
        pol = TunedHeuristic(**tuned[spec.split(":", 1)[1]]["params"])
    else:
        m, c = load(spec.split(":", 1)[1]); pol = LearnedPolicy(m, c, "gnn")
    return Guard(pol, guard) if guard else pol


RUNS = [("ecmp", "ideal"), ("tuned:snmp", "snmp"), ("tuned:degraded", "degraded"), (M + ".pt".replace(".pt", ""), "snmp")]
RUNS = [("ecmp", "ideal"), ("tuned:snmp", "snmp"), ("tuned:degraded", "degraded"), ("model:runs/ours_dfm_final.pt", "snmp"),
        ("model:runs/ours_dfm_final.pt", "degraded"), ("guardL2-0.06@model:runs/ours_dfm_final.pt", "degraded")]
rows = []
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    sc_file = d / "scenario.json"
    if not sc_file.exists():
        continue
    sc = S.load(sc_file)
    runs = [("model:runs/ours_dfm_final.pt", "degraded")] if d.name.endswith("_h0.2") else RUNS
    for r in A.compare(d, sc, runs, make_policy):
        r["scenario"], r["hyst"] = d.name.replace("_h0.2", ""), 0.2 if d.name.endswith("_h0.2") else 0.05
        rows.append(r)
json.dump(rows, open("experiments/final/emu2_rows.json", "w"), indent=1, default=float)
lab = lambda r: r["spec"].replace("model:runs/", "").replace(".pt", "") + f" [{r['profile']}]" + (" hyst0.2" if r["hyst"] == 0.2 else "")
by = defaultdict(list)
for r in rows:
    by[lab(r)].append(r)
print("| policy [telemetry] | scenarios | emu max-util | sim max-util | time-series Pearson | emu loss | flow-mods/min | ctrl loop ms |")
print("|---|---|---|---|---|---|---|---|")
f = lambda k, rs: np.mean([r[k] for r in rs])
for k, rs in by.items():
    print(f"| {k} | {len(rs)} | {f('emu_mu', rs):.3f} | {f('sim_mu', rs):.3f} | {f('pearson', rs):.2f} | {f('emu_loss', rs):.4f} | {f('flowmods_per_min', rs):.1f} | {f('latency_ms', rs):.0f} |")
e = np.array([r["emu_mu"] for r in rows if r["hyst"] == 0.05]); s = np.array([r["sim_mu"] for r in rows if r["hyst"] == 0.05])
print(f"\nover {len(e)} runs: Pearson(emu, sim mean max-util) = {stats.pearsonr(e, s)[0]:.2f}; mean |emu - sim| = {np.abs(e - s).mean():.3f}; "
      f"bias (emu - sim) = {np.mean(e - s):+.3f}")
# does the simulator rank the policies like the emulator, per scenario?
agree = []
for sc in sorted({r["scenario"] for r in rows}):
    rs = [r for r in rows if r["scenario"] == sc and r["hyst"] == 0.05]
    if len(rs) > 3:
        agree.append(stats.kendalltau([r["emu_mu"] for r in rs], [r["sim_mu"] for r in rs])[0])
print("Kendall tau of policy ranking (emu vs sim) per scenario:", [round(a, 2) for a in agree])
