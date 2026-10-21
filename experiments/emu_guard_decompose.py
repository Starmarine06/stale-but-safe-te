"""Where does the guard's extra max-utilisation in the TCP emulation come from?
Per interval, the guarded run's true max-util is compared with the unguarded learned run and the ECMP run of the same scenario
(same demand schedule). Excess = guard - learned is split into
  forgone : fallback intervals, (ECMP - learned)   -- the learned gain given up by falling back
  other   : everything else (state divergence, switching transients, noise)
and fallback intervals are split by whether the learned policy was actually ahead of ECMP in that interval.
  python experiments/emu_guard_decompose.py"""
import json
from pathlib import Path
import numpy as np
from stale_but_safe.emu import scenario as S
from stale_but_safe.emu.run_emulation import tag_of

N = ["s1", "s2", "s3", "n14", "n15", "n16", "n17", "n18", "n19"]
SKIP = 3


def mo(n):
    p = Path("runs/emu_tcp") / n / "model.txt"
    return p.read_text().strip() if p.exists() else f"runs/ours_dfm_final_{n}.pt"


def rows(root, n, spec, prof):
    d = Path(root) / n; tag = tag_of(spec, prof)
    r = json.loads((d / tag / f"ctrl_{tag}.json").read_text())["rows"]
    return {x["interval"]: x for x in r if x["interval"] >= SKIP}


def decompose(label, groot, gspec, lroot):
    tot = dict(excess=[], forgone=[], other=[], fb=[], fb_good=[], fb_bad=[], trans_ex=[], steady_ex=[])
    per = []
    for n in N:
        M = mo(n)
        g = rows(groot, n, gspec(M), "degraded")
        l = rows(lroot, n, f"model:{M}", "degraded")
        e = rows("runs/emu_tcp", n, "ecmp", "ideal")
        ks = sorted(set(g) & set(l) & set(e))
        gm = np.array([g[k]["true_max_util"] for k in ks]); lm = np.array([l[k]["true_max_util"] for k in ks])
        em = np.array([e[k]["true_max_util"] for k in ks]); fb = np.array([g[k]["fallback"] for k in ks], bool)
        ex = gm - lm
        forgone = np.where(fb, em - lm, 0.0)
        prev = np.r_[fb[0], fb[:-1]]; trans = fb != prev
        tot["excess"].append(ex.mean()); tot["forgone"].append(forgone.mean()); tot["other"].append((ex - forgone).mean())
        tot["fb"].append(fb.mean())
        tot["fb_good"].append((fb & (lm < em - 0.01)).sum() / max(fb.sum(), 1))   # learned clearly ahead
        tot["fb_bad"].append((fb & (lm > em + 0.01)).sum() / max(fb.sum(), 1))    # learned clearly behind
        tot["trans_ex"].append(ex[trans].mean() if trans.any() else np.nan)
        tot["steady_ex"].append(ex[~trans].mean())
        per.append((n, fb.mean(), ex.mean(), forgone.mean()))
    f = lambda k: np.nanmean(tot[k])
    print(f"| {label} | {100*f('fb'):.0f}% | {f('excess'):+.4f} | {f('forgone'):+.4f} | {f('other'):+.4f} "
          f"| {100*f('fb_good'):.0f}% | {100*f('fb_bad'):.0f}% | {f('trans_ex'):+.3f} | {f('steady_ex'):+.4f} |")
    return per


print("| guard run vs matching unguarded learned | fallback | excess (guard-learned) | forgone gain | other | fb where learned ahead | fb where learned behind | excess at switch intervals | excess elsewhere |")
print("|---|---|---|---|---|---|---|---|---|")
a = decompose("guard0.05, h0.05", "runs/emu_tcp", lambda M: f"guard0.05@model:{M}", "runs/emu_tcp")
b = decompose("guardL2-0.06, h0.05", "runs/emu_tcp_var", lambda M: f"guardL2-0.06@model:{M}", "runs/emu_tcp")
c = decompose("guard0.05, h0.2", "runs/emu_tcp_gh", lambda M: f"guard0.05@model:{M}", "runs/emu_tcp_varh")
d = decompose("guardL2-0.06, h0.2", "runs/emu_tcp_gh", lambda M: f"guardL2-0.06@model:{M}", "runs/emu_tcp_varh")
print("\nper scenario (guard0.05, h0.05): scenario, fallback share, excess, forgone")
for r in a: print("  %-4s %4.0f%% %+.4f %+.4f" % (r[0], 100 * r[1], r[2], r[3]))
