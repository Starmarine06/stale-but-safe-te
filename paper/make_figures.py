"""Figures for the paper, drawn only from files in experiments/final/. Run from the repository root: python paper/make_figures.py"""
import glob, json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

F = "experiments/final/"; OUT = "paper/figures/"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42, "legend.fontsize": 8, "font.family": "DejaVu Sans"})
LEARN, HEUR, OLD = "#1b6ca8", "#d9822b", "#9aa5b1"
rng = np.random.default_rng(0)
def save(fig, name):
    fig.savefig(OUT + name + ".png"); fig.savefig(OUT + name + ".pdf")
def boot(x, n=4000):
    x = np.asarray(x); b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]; return np.percentile(b, [2.5, 97.5])

# ---------- Fig 1: Zoo dScore and p99 regret by telemetry profile
raw = json.load(open(F + "retune_profiles.json"))
L = defaultdict(lambda: defaultdict(list)); old = defaultdict(dict)
for p in [F + "results_ours.json"] + sorted(glob.glob(F + "results_seed*.json")):
    for r in json.load(open(p))["results"]:
        if r["method"] == "gnn-ours": L[r["profile"]][r["topo"]].append(r)
for r in json.load(open(F + "results_ours.json"))["results"]:
    if r["method"] == "tuned": old[r["profile"]][r["topo"]] = r
profs = ["ideal", "rich", "snmp", "degraded"]; lab = ["ideal", "rich", "SNMP", "degraded"]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
w = 0.38; x = np.arange(4)
for k, (name, col) in enumerate([("learned", LEARN), ("re-tuned heuristic", HEUR)]):
    for j, pr in enumerate(profs):
        topos = sorted(L[pr])
        if name == "learned":
            d = np.array([np.mean([r["delta"] for r in L[pr][t]]) for t in topos]); p = np.array([np.mean([r["regret_p99"] for r in L[pr][t]]) for t in topos])
        elif name.startswith("original"):
            d = np.array([old[pr][t]["delta"] for t in topos]); p = np.array([old[pr][t]["regret_p99"] for t in topos])
        else:
            rs = [r for r in raw if r["profile"] == pr]
            d = np.mean([[r["per_topo"][t]["delta"] for t in topos] for r in rs], 0); p = np.mean([[r["per_topo"][t]["regret_p99"] for t in topos] for r in rs], 0)
        for a, v in ((ax[0], d), (ax[1], p)):
            lo, hi = boot(v)
            a.bar(x[j] + (k - 0.5) * w, v.mean(), w, color=col, label=name if j == 0 else None, yerr=[[v.mean() - lo], [hi - v.mean()]], capsize=2, error_kw={"lw": 0.8})
ax[0].axhline(0, color="k", lw=0.6); ax[0].set_ylabel("dScore vs ECMP (lower = better)"); ax[1].set_ylabel("p99 regret vs ECMP (lower = better)")
for a in ax: a.set_xticks(x); a.set_xticklabels(lab); a.set_xlabel("telemetry profile")
h, l_ = ax[0].get_legend_handles_labels(); fig.legend(h, l_, loc="upper center", ncol=2, frameon=False, fontsize=8.5); ax[0].set_title("(a) mean score", fontsize=9); ax[1].set_title("(b) tail risk", fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.92)); save(fig, "fig1_zoo"); plt.close(fig)

# ---------- Fig 2: real traces, learned minus re-tuned heuristic, window-level CIs
cells = {}
for f in sorted(glob.glob(F + "realci_v2_*.json")):
    o = json.load(open(f))
    for key, cell in o["cells"].items():
        ld, pn = key.split("|"); t = np.array(cell["tuned"]) * 100
        l = np.mean([cell[k] for k in cell if k.startswith("learned")], 0) * 100; ok = np.isfinite(t) & np.isfinite(l); d = (l - t)[ok]
        lo, hi = boot(d, 3000); cells[(o["network"], float(ld), pn)] = (d.mean(), lo, hi)
nets = ["abilene", "geant", "germany17"]; rows = [(n, ld) for n in nets for ld in (0.6, 0.8, 1.0)]
M = np.array([[cells[(n, ld, p)][0] for p in profs] for n, ld in rows])
fig, ax = plt.subplots(figsize=(4.6, 3.6))
im = ax.imshow(M, cmap="RdBu", vmin=-13, vmax=13, aspect="auto")
for i, (n, ld) in enumerate(rows):
    for j, p in enumerate(profs):
        m, lo, hi = cells[(n, ld, p)]; mark = "L" if lo > 0 else "H" if hi < 0 else "="
        ax.text(j, i, f"{m:+.1f}\n{mark}", ha="center", va="center", fontsize=7, color="white" if abs(m) > 8 else "black")
ax.set_xticks(range(4)); ax.set_xticklabels(lab); ax.set_yticks(range(9)); ax.set_yticklabels([f"{n} @{ld}" for n, ld in rows])
ax.set_title("Learned minus re-tuned heuristic, gain points\nblue = learned ahead (L), red = heuristic ahead (H), = tie", fontsize=8)
fig.colorbar(im, ax=ax, shrink=0.8, label="points of gain"); fig.tight_layout(); save(fig, "fig2_real"); plt.close(fig)

# ---------- Fig 3: delay x polling grid with window-level CIs
ci = json.load(open(F + "delay_ci.json"))
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
for a, net in zip(axs, ["abilene", "geant"]):
    for P, col in zip((1, 5, 20), ("#1b6ca8", "#3b9a6a", "#a8421b")):
        xs, ms, los, his = [], [], [], []
        for d in (0, 3, 10):
            r = next(x for x in ci if x["d"] == d and x["P"] == P)["net"][net]
            df = 100 * (np.array(r["learned"]) - np.array(r["heur"])); lo, hi = boot(df, 3000)
            xs.append(d + {1: -0.25, 5: 0, 20: 0.25}[P]); ms.append(df.mean()); los.append(df.mean() - lo); his.append(hi - df.mean())
        a.errorbar(xs, ms, yerr=[los, his], fmt="o-", color=col, capsize=2, lw=1, ms=3.5, label=f"polling P={P}")
    a.axhline(0, color="k", lw=0.6); a.set_xticks([0, 3, 10]); a.set_xlabel("telemetry delay d (intervals)"); a.set_title({"abilene": "Abilene", "geant": "GEANT"}[net], fontsize=9)
axs[0].set_ylabel("learned minus heuristic (gain points)")
h, l_ = axs[0].get_legend_handles_labels(); fig.legend(h, l_, loc="upper center", ncol=3, frameon=False, fontsize=8.5)
fig.tight_layout(rect=(0, 0, 1, 0.92)); save(fig, "fig3_delay"); plt.close(fig)
print("figures written")
