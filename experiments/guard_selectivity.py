"""NOTE: the simulator rows use same-state counterfactual labels and the emulation row uses labels from separate runs; they are not comparable (simulator with separate-run labels: 0.38 vs 0.34). Kept for reference only.
Does the guard fall back when the learned policy is NOT ahead of ECMP? Same scenarios, simulator replay vs TCP emulation.
'ahead' = the learned policy's max-utilisation is more than 0.01 below ECMP's in that interval (simulator: true fluid
utilisation of the raw proposal, uncapped and capped at 1 like emulator counters; emulation: measured max-utilisation of the
unguarded learned run vs the ECMP run).  Guard: guard0.05 (eps 0.05, all layers), controller update threshold 0.05.
  python experiments/guard_selectivity.py"""
import json
import sys
from pathlib import Path
import numpy as np

sys.argv = ["x"]
src = open("experiments/guard_carried_replay.py", encoding="utf-8").read().split('print("| guard | counters')[0]
exec(src)
from stale_but_safe.emu.run_emulation import tag_of


def stats(ahead, fb):
    return fb[ahead].mean(), fb[~ahead].mean(), fb.mean(), ahead.mean()


def emu_rows(root, n, spec, prof):
    t = tag_of(spec, prof); r = json.loads((Path(root) / n / t / f"ctrl_{t}.json").read_text())["rows"]
    return {x["interval"]: x for x in r if x["interval"] >= 3 and x["latency_ms"] > 0}


out = {"sim uncapped": [], "sim capped": [], "emulation": []}
for n in NAMES:
    R = run(n, True, GUARDS["guard0.05"]); raw, em, fb = R[:, 0], R[:, 2], R[:, 3].astype(bool)
    out["sim uncapped"].append(stats(raw < em - .01, fb))
    out["sim capped"].append(stats(np.minimum(raw, 1) < np.minimum(em, 1) - .01, fb))
    M = (Path("runs/emu_tcp") / n / "model.txt").read_text().strip() if (Path("runs/emu_tcp") / n / "model.txt").exists() else f"runs/ours_dfm_final_{n}.pt"
    g, l, e = emu_rows("runs/emu_tcp", n, f"guard0.05@model:{M}", "degraded"), emu_rows("runs/emu_tcp", n, f"model:{M}", "degraded"), emu_rows("runs/emu_tcp", n, "ecmp", "ideal")
    ks = sorted(set(g) & set(l) & set(e))
    lm = np.array([l[k]["true_max_util"] for k in ks]); em_ = np.array([e[k]["true_max_util"] for k in ks]); fb_ = np.array([g[k]["fallback"] for k in ks], bool)
    out["emulation"].append(stats(lm < em_ - .01, fb_))
print("| setting | P(fallback \| learned ahead) | P(fallback \| not ahead) | fallback share | share of intervals learned is ahead |\n|---|---|---|---|---|")
for k, v in out.items():
    a = np.array(v)
    print(f"| {k} | {np.nanmean(a[:,0]):.2f} | {np.nanmean(a[:,1]):.2f} | {np.nanmean(a[:,2]):.2f} | {np.nanmean(a[:,3]):.2f} |")
