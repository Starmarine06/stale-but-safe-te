"""Emulation with the re-tuned heuristic (runs/emu3, same scenarios as runs/emu2): re-tuned vs original heuristic vs learned (from emu2_rows.json)."""
import json
from pathlib import Path
import numpy as np, yaml
from scipy import stats
from stale_but_safe.baselines.heuristics import TunedHeuristic
from stale_but_safe.emu import analyze as A, scenario as S
v2 = yaml.safe_load(open("configs/tuned_heuristic_v2.yaml"))
def mk(spec, prof):
    return TunedHeuristic(**v2[spec.split(":", 1)[1]]["params"])
rows = []
for d in sorted(p for p in Path("runs/emu3").iterdir() if p.is_dir()):
    sc = S.load(d / "scenario.json")
    for r in A.compare(d, sc, [("tunedv2:snmp", "snmp"), ("tunedv2:degraded", "degraded")], mk):
        r["scenario"] = d.name; rows.append(r)
json.dump(rows, open("experiments/final/emu3_rows.json", "w"), indent=1, default=float)
old = json.load(open("experiments/final/emu2_rows.json"))
def sel(rs, spec, prof): return [r for r in rs if r["spec"] == spec and r["profile"] == prof and r.get("hyst", 0.05) == 0.05]
print("Emulation, 4 scenarios (emu10 x3 seeds, er12), mean over scenarios\n")
print("| policy [telemetry] | emu max-util | sim max-util | emu loss | flow-mods/min | ctrl loop ms |\n|---|---|---|---|---|---|")
f = lambda rs, k: np.mean([r[k] for r in rs])
for name, rs in [("original heuristic [snmp]", sel(old, "tuned:snmp", "snmp")), ("re-tuned heuristic [snmp]", sel(rows, "tunedv2:snmp", "snmp")),
                 ("learned [snmp]", sel(old, "model:runs/ours_dfm_final.pt", "snmp")),
                 ("original heuristic [degraded]", sel(old, "tuned:degraded", "degraded")), ("re-tuned heuristic [degraded]", sel(rows, "tunedv2:degraded", "degraded")),
                 ("learned [degraded]", sel(old, "model:runs/ours_dfm_final.pt", "degraded")), ("ECMP", sel(old, "ecmp", "ideal"))]:
    print(f"| {name} | {f(rs,'emu_mu'):.3f} | {f(rs,'sim_mu'):.3f} | {f(rs,'emu_loss'):.4f} | {f(rs,'flowmods_per_min'):.1f} | {f(rs,'latency_ms'):.0f} |")
e = np.array([r["emu_mu"] for r in rows]); s = np.array([r["sim_mu"] for r in rows])
print(f"\nre-tuned heuristic runs (n={len(e)}): Pearson(emu, sim mean max-util) = {stats.pearsonr(e, s)[0]:.2f}; mean |emu - sim| = {np.abs(e - s).mean():.3f}")
print("\nPer scenario, emulated mean max-util (lower = better):\n\n| scenario | ECMP | re-tuned snmp | learned snmp | re-tuned degraded | learned degraded |\n|---|---|---|---|---|---|")
for sc in sorted({r["scenario"] for r in rows}):
    g = lambda rs, sp, pf: [r["emu_mu"] for r in rs if r["scenario"] == sc and r["spec"] == sp and r["profile"] == pf and r.get("hyst", 0.05) == 0.05][0]
    print(f"| {sc} | {g(old,'ecmp','ideal'):.3f} | {g(rows,'tunedv2:snmp','snmp'):.3f} | {g(old,'model:runs/ours_dfm_final.pt','snmp'):.3f} | {g(rows,'tunedv2:degraded','degraded'):.3f} | {g(old,'model:runs/ours_dfm_final.pt','degraded'):.3f} |")
