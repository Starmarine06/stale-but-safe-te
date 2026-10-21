"""Which guard signal is wrong in the emulator? Uses the guard_trace logged by the instrumented controller (runs/emu_tcp_diag).
For each guarded run: layer that fired, and the guard's own estimates against what the unguarded learned run and the ECMP run
(same scenario and demand schedule) actually measured in the same interval.
  python experiments/emu_guard_trace.py"""
import json
from pathlib import Path
import numpy as np
from stale_but_safe.emu.run_emulation import tag_of

SKIP = 3


def load(d, spec, prof):
    t = tag_of(spec, prof); c = json.loads((d / t / f"ctrl_{t}.json").read_text())
    rows = [r for r in c["rows"] if r["latency_ms"] > 0]
    return rows, c.get("guard_trace"), c["guard"]


def by_interval(rows):
    return {r["interval"]: r for r in rows}


for n in ["s2", "n16"]:
    d = Path("runs/emu_tcp_diag") / n
    M = f"runs/ours_dfm_final_{n}.pt"
    mp = Path("runs/emu_tcp") / n / "model.txt"
    if mp.exists(): M = mp.read_text().strip()
    ecmp = by_interval(load(d, "ecmp", "ideal")[0]); lrn = by_interval(load(d, f"model:{M}", "degraded")[0])
    for gname in (f"guard0.05@model:{M}", f"guardL2-0.06@model:{M}"):
        rows, tr, G = load(d, gname, "degraded")
        assert tr is not None and len(tr) == len(rows), (len(tr or []), len(rows))
        print(f"\n== {n} {gname.split('@')[0]}  stats={G}")
        A = []
        for r, t in zip(rows, tr):
            k = r["interval"]
            if k < SKIP or k not in ecmp or k not in lrn: continue
            A.append((k, t[0], t[1], t[2], t[3], t[4], t[5], r["obs_max_util"], r["true_max_util"], ecmp[k]["true_max_util"], lrn[k]["true_max_util"]))
        A = np.array(A, float)
        lay, p, e, ex, meas, cf, obsmu, gmu, emu, lmu = (A[:, i] for i in range(1, 11))
        print(f"layer share: none {np.mean(lay==0):.2f}  L1 {np.mean(lay==1):.2f}  L2 {np.mean(lay==2):.2f}  L3 {np.mean(lay==3):.2f}")
        ok = ~np.isnan(e)
        print(f"L2 inputs (when evaluated, n={ok.sum()}): predicted proposal/ECMP mu on estimated demand: {np.mean(p[ok]):.3f}/{np.mean(e[ok]):.3f}; "
              f"true learned/ECMP mu: {np.mean(lmu[ok]):.3f}/{np.mean(emu[ok]):.3f}")
        print(f"  predicted ratio p/e: median {np.median(p[ok]/e[ok]):.3f}  | true ratio learned/ECMP: median {np.median(lmu[ok]/emu[ok]):.3f}")
        print(f"  corr(predicted p-e, true learned-ECMP) = {np.corrcoef((p-e)[ok], (lmu-emu)[ok])[0,1]:+.2f}")
        okc = ~np.isnan(cf)
        print(f"L3 inputs (n={okc.sum()}): observed max util {np.mean(meas[okc]):.3f} (true {np.mean(gmu[okc]):.3f}); "
              f"counterfactual ECMP mu from estimated demand {np.mean(cf[okc]):.3f} (true ECMP {np.mean(emu[okc]):.3f}); mean excess {np.mean(ex[okc]):+.3f} vs tau 0.05")
        print(f"  corr(cf, true ECMP mu) = {np.corrcoef(cf[okc], emu[okc])[0,1]:+.2f}   corr(observed mu, true mu) = {np.corrcoef(meas[okc], gmu[okc])[0,1]:+.2f}")
        fb = lay > 0
        print(f"  learned ahead of ECMP (by >0.01) in {np.mean(lmu < emu - .01):.2f} of intervals; fallbacks land there {np.mean((lmu < emu - .01)[fb]):.2f} of the time")
