"""TE audit: "is traffic engineering worth it on my network, and does it survive my telemetry?"

  stale-but-safe audit --network abilene|geant|germany17|<ZooName>|path/to.graphml
                       [--traffic trace.npz|trace.csv] [--polling 5 --delay 1 --loss 0.02 --noise 0.05]
                       [--out report] [--load 0.8] [--model runs/ours_dfm_final.pt]

Writes <out>.json and a self-contained <out>.html. Everything is measured in the fluid simulator on
paired seeds; the verdict is a rule over those measurements (see `verdict`).
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import yaml

from .baselines.heuristics import ECMP, LPOracle, TunedHeuristic
from .env import real
from .env.te_env import EnvCfg
from .env.telemetry import PROFILES, TelemetryCfg
from .eval.compare import episode_stats
from .guard.guard import Guard, GuardCfg

ROOT = Path(__file__).resolve().parents[2]
GUARD = GuardCfg(eps=-0.06, layers=(True, True, False))   # L2-only, chosen on validation (see paper)


# ------------------------------------------------------------------ input
def load_network(spec: str, k: int = 4):
    """-> (Topology, node labels or None, built-in real trace or None)."""
    if spec in real.NETS:
        topo, labels = real.build_topology(spec, k)
        return topo, labels, real.load(spec, k)[1]
    if spec.lower().endswith(".graphml"):
        import networkx as nx
        from .env.topology import degree_capacities, from_graph
        from .env.zoo import load_graph
        g = load_graph(spec)
        if g is None:
            raise SystemExit(f"cannot read {spec}")
        labels = [str(n) for n in g.nodes]
        return from_graph(g, k=k, name=Path(spec).stem), labels, None
    from .cli import get_topology
    return get_topology(spec), None, None


def load_trace(path: str, topo, labels) -> np.ndarray:
    """npz (key 'trace', shape T x D in topology demand order) or long CSV `t,src,dst,value`
    (src/dst = node names for built-in/graphml networks, else node indices)."""
    if path.endswith(".npz"):
        return np.load(path)["trace"].astype(float)
    import csv
    idx = {n: i for i, n in enumerate(labels)} if labels else {}
    pos = {(s, t): j for j, (s, t) in enumerate(topo.demands)}
    rows = {}
    for r in csv.DictReader(open(path)):
        s = idx.get(r["src"], None) if idx else int(r["src"])
        t = idx.get(r["dst"], None) if idx else int(r["dst"])
        if s is None or t is None or s == t:
            continue
        rows.setdefault(r["t"], np.zeros(topo.D))[pos[(s, t)]] += float(r["value"])
    if not rows:
        raise SystemExit("traffic CSV had no usable rows")
    return np.stack([rows[k] for k in sorted(rows, key=lambda x: float(x) if x.replace('.', '', 1).isdigit() else x)])


# ------------------------------------------------------------------ measurement
def _tuned_profile(c: TelemetryCfg) -> str:
    sev = (c.P - 1) + c.d + 10 * c.p + 5 * c.sigma
    return "ideal" if sev < 0.5 else "rich" if sev < 2.5 else "snmp" if sev < 6 else "degraded"


def evaluate(topo, cfg, pol, seeds):
    """One rollout per seed (= one trace window); keeps per-window gains so uncertainty can be reported."""
    rows = [episode_stats(topo, cfg, pol, s) for s in seeds]
    reg = np.concatenate([r["regret"] for r in rows])
    gains = np.array([1 - r["mu"] / r["ecmp_mu"] for r in rows])
    return {"improvement_vs_ecmp": float(gains.mean()), "regret_p99": float(np.quantile(reg, 0.99)),
            "raw": {"regret": reg}, "mean_loss": float(np.mean([r["loss"] for r in rows])), "gains": gains}


def _ci(g, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(g, (n, len(g))).mean(1) if len(g) > 1 else np.array([g.mean()])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _summ(r):
    reg, g = r["raw"]["regret"], r["gains"]
    lo, hi = _ci(g)
    pos = np.clip(g, 0, None)
    return {"gain_pct": 100 * r["improvement_vs_ecmp"], "gain_ci_pct": [100 * lo, 100 * hi], "windows": int(len(g)),
            "top_window_share": float(pos.max() / pos.sum()) if pos.sum() > 1e-9 else 0.0,
            "p99_regret": r["regret_p99"], "worse_than_ecmp_pct": 100 * float((reg > 0.01).mean()), "loss": r["mean_loss"]}


OOD = {"P": 10, "d": 5, "sigma": 0.3, "p": 0.2}      # largest values covered by the training/evaluation grid


def ood_notes(t: TelemetryCfg) -> list[str]:
    return [f"telemetry {k}={getattr(t, k)} is outside the range this tool was evaluated on (max {v}); treat results as extrapolation"
            for k, v in OOD.items() if getattr(t, k) > v]


def audit(topo, trace, user_tele: TelemetryCfg, load: float, horizon: int, seeds: int, model: str | None):
    tuned = yaml.safe_load(open(ROOT / "configs" / "tuned_heuristic.yaml"))
    learned = None
    if model and Path(model).exists():
        from .models.learned import LearnedPolicy, load as load_model
        m, c = load_model(model)
        learned = (m, c, LearnedPolicy)
    profs = {"ideal": PROFILES["ideal"], "snmp": PROFILES["snmp"], "degraded": PROFILES["degraded"],
             "yours": user_tele}
    out = {"seeds": seeds, "ood": ood_notes(user_tele), "network": topo.name, "nodes": topo.n, "links": topo.E, "demands": topo.D,
           "traffic": "real trace" if trace is not None else "synthetic gravity+bursts",
           "load": load, "user_telemetry": vars(user_tele), "results": {}, "headroom": None}
    for pn, tele in profs.items():
        cfg = EnvCfg(load=load, horizon=horizon, telemetry=tele, trace=trace)
        pols = {"ecmp": ECMP(), "tuned": TunedHeuristic(**tuned[_tuned_profile(tele)]["params"])}
        if learned:
            pols["learned"] = learned[2](learned[0], learned[1], "gnn")
            pols["learned+guard"] = Guard(learned[2](learned[0], learned[1], "gnn"), GUARD)
        pols["tuned+guard"] = Guard(TunedHeuristic(**tuned[_tuned_profile(tele)]["params"]), GUARD)
        out["results"][pn] = {k: _summ(evaluate(topo, cfg, p, range(seeds))) for k, p in pols.items()}
        if pn == "ideal" and topo.n <= 25:
            out["headroom"] = _summ(evaluate(topo, cfg, LPOracle(), range(seeds)))
    # polling sweep with the best non-oracle method on the user's profile
    best = max((k for k in out["results"]["yours"] if k != "ecmp"), key=lambda k: out["results"]["yours"][k]["gain_pct"])
    out["polling_sweep"] = {"method": best, "rows": {}}
    for P in (1, 2, 5, 10):
        t = TelemetryCfg(P, user_tele.d, user_tele.sigma, user_tele.p)
        cfg = EnvCfg(load=load, horizon=horizon, telemetry=t, trace=trace)
        pol = {"tuned": TunedHeuristic(**tuned[_tuned_profile(t)]["params"]),
               "tuned+guard": Guard(TunedHeuristic(**tuned[_tuned_profile(t)]["params"]), GUARD)}.get(best)
        if pol is None and learned:
            from .models.learned import LearnedPolicy
            pol = LearnedPolicy(learned[0], learned[1], "gnn")
            pol = Guard(pol, GUARD) if best == "learned+guard" else pol
        out["polling_sweep"]["rows"][str(P)] = _summ(evaluate(topo, cfg, pol, range(seeds)))
    out["verdict"] = verdict(out)
    return out


def verdict(o: dict) -> dict:
    """Offline estimate, not a deployment recommendation. Outcomes: 'Stay on ECMP', 'Use <method>' or
    'Insufficient evidence' (too few windows, gain interval straddles the 5% threshold, or gain concentrated)."""
    ideal, yours = o["results"]["ideal"], o["results"]["yours"]
    exact = o["headroom"] is not None
    lines = list(o.get("ood", []))
    best_ideal = max((k for k in ideal if k != "ecmp"), key=lambda k: ideal[k]["gain_pct"])
    bi = ideal[best_ideal]
    head = o["headroom"]["gain_pct"] if exact else bi["gain_pct"]
    lo, hi = bi["gain_ci_pct"]
    if bi["windows"] < 5:
        return {"action": "Insufficient evidence.", "reasons": lines + [
            f"Only {bi['windows']} trace windows were evaluated; at least 5 are needed for an interval estimate."]}
    if hi < 5:
        who = "Even an all-knowing optimiser" if exact and head < 5 else "No method we tested, even with perfect telemetry,"
        return {"action": "Stay on ECMP.", "reasons": lines + [
            f"{who} gains only {bi['gain_pct']:.1f}% max-utilisation here (95% interval over {bi['windows']} windows "
            f"[{lo:.1f}, {hi:.1f}]), so a TE controller is not worth its operational risk."]}
    if lo < 5:
        return {"action": "Insufficient evidence.", "reasons": lines + [
            f"The best method's gain with perfect telemetry is {bi['gain_pct']:.1f}% with a 95% interval [{lo:.1f}, {hi:.1f}] over "
            f"{bi['windows']} windows, which straddles the 5% threshold; benefits depend on which traffic windows occur.",
            f"The single best window supplies {100 * bi['top_window_share']:.0f}% of the total gain. Collect more traffic or run in shadow mode."]}
    cand = {k: v for k, v in yours.items() if k != "ecmp"}
    safe = {k: v for k, v in cand.items() if v["p99_regret"] <= 0.02}
    pool = safe or cand
    pick = max(pool, key=lambda k: pool[k]["gain_pct"])
    what = (f"Best-case headroom (LP oracle) is {head:.1f}%" if exact else
            f"The best method tested gains {bi['gain_pct']:.1f}% with perfect telemetry (exact optimum not computed above 25 nodes)")
    pl, ph = pool[pick]["gain_ci_pct"]
    lines.append(f"{what}; with your telemetry {pick} captures {pool[pick]['gain_pct']:.1f}% (95% interval [{pl:.1f}, {ph:.1f}]).")
    t = yours["tuned"]
    if t["p99_regret"] > 0.05 or t["worse_than_ecmp_pct"] > 15:
        lines.append(f"The tuned heuristic is fragile under your telemetry: it is worse than ECMP in "
                     f"{t['worse_than_ecmp_pct']:.0f}% of intervals (p99 regret {t['p99_regret']:+.2f}).")
    if not safe:
        lines.append("No method keeps p99 regret under 0.02; run in shadow mode before any rollout.")
    if pool[pick]["top_window_share"] > 0.5 and pool[pick]["windows"] >= 5:
        lines.append(f"Warning: {100 * pool[pick]['top_window_share']:.0f}% of {pick}'s gain comes from one window.")
    rows = o["polling_sweep"]["rows"]
    g = {p: rows[p]["gain_pct"] for p in rows}
    bestP, spread = max(g, key=g.get), max(g.values()) - min(g.values())
    if spread < 2:
        lines.append(f"Polling rate barely matters for {o['polling_sweep']['method']} here "
                     f"(gain varies only {spread:.1f} points between 1 and 10 intervals).")
    else:
        lines.append(f"Polling every {bestP} interval(s) works best for {o['polling_sweep']['method']} "
                     f"({g[bestP]:.1f}% vs {g[min(g, key=g.get)]:.1f}% at the worst rate).")
    return {"action": f"Use {pick}.", "reasons": lines}


# ------------------------------------------------------------------ report
def to_html(o: dict) -> str:
    esc = html.escape
    names = ["ecmp", "tuned", "tuned+guard", "learned", "learned+guard"]
    profs = list(o["results"])
    cols = {"ecmp": "#8a8f98", "tuned": "#d98e04", "tuned+guard": "#e8c15a", "learned": "#2f6fdd", "learned+guard": "#7aa7f5"}
    gmax = max([1.0] + [v["gain_pct"] for p in o["results"].values() for v in p.values()])
    W, H, pad = 640, 220, 34
    bw = (W - 2 * pad) / len(profs)
    svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Gain over ECMP by telemetry profile" class="chart">']
    for gi, pn in enumerate(profs):
        present = [n for n in names if n in o["results"][pn]]
        w = bw * 0.8 / max(len(present), 1)
        for bi, n in enumerate(present):
            v = max(0.0, o["results"][pn][n]["gain_pct"])
            h = (H - 2 * pad) * v / gmax
            x = pad + gi * bw + bw * 0.1 + bi * w
            svg.append(f'<rect x="{x:.1f}" y="{H - pad - h:.1f}" width="{w - 2:.1f}" height="{h:.1f}" fill="{cols[n]}"><title>{n} {pn}: {v:.1f}%</title></rect>')
            if n != "ecmp":
                svg.append(f'<text x="{x + (w - 2) / 2:.1f}" y="{H - pad - h - 3:.1f}" text-anchor="middle" font-size="9">{v:.0f}</text>')
        svg.append(f'<text x="{pad + gi * bw + bw / 2:.1f}" y="{H - 12}" text-anchor="middle">{esc(pn)}</text>')
    svg.append(f'<text x="4" y="14">% max-util gain vs ECMP (max {gmax:.0f})</text></svg>')
    legend = "".join(f'<span><i style="background:{cols[n]}"></i>{n}</span>' for n in names if n in o["results"]["ideal"])
    rows = []
    for pn, res in o["results"].items():
        for n, v in res.items():
            rows.append(f"<tr><td>{esc(pn)}</td><td>{esc(n)}</td><td>{v['gain_pct']:.1f}</td><td>{v['p99_regret']:+.3f}</td>"
                        f"<td>{v['worse_than_ecmp_pct']:.0f}</td></tr>")
    ps = "".join(f"<tr><td>{p}</td><td>{v['gain_pct']:.1f}</td><td>{v['p99_regret']:+.3f}</td></tr>"
                 for p, v in o["polling_sweep"]["rows"].items())
    head = o["headroom"]
    hh = f"{head['gain_pct']:.1f}%" if head else "n/a (network too large for the exact LP)"
    vd = o["verdict"]
    u = o["user_telemetry"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TE audit - {esc(o['network'])}</title><style>
:root{{--bg:#fff;--fg:#1c1f24;--mut:#5d6570;--card:#f4f6f9;--line:#d9dee5;--acc:#2f6fdd}}
@media (prefers-color-scheme:dark){{:root{{--bg:#14171b;--fg:#e8eaed;--mut:#98a0ab;--card:#1d2127;--line:#2e343c;--acc:#7aa7f5}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:820px;margin:0 auto;padding:24px 16px 48px}} h1{{font-size:24px;margin:0 0 4px}} h2{{font-size:17px;margin:28px 0 8px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}} .mut{{color:var(--mut)}}
.verdict{{font-size:20px;font-weight:650;color:var(--acc)}} table{{border-collapse:collapse;width:100%;font-size:14px}}
td,th{{padding:5px 8px;border-bottom:1px solid var(--line);text-align:left}} .chart{{width:100%;height:auto;fill:var(--mut);font-size:11px}}
.legend span{{margin-right:14px}} .legend i{{display:inline-block;width:10px;height:10px;margin-right:5px;border-radius:2px}}
</style></head><body><main>
<h1>TE audit: {esc(o['network'])}</h1>
<p class="mut">{o['nodes']} nodes, {o['links']} links, {o['demands']} demands - traffic: {esc(o['traffic'])} (scaled to median ECMP max-util {o['load']}) -
your telemetry: poll every {u['P']} interval(s), delay {u['d']}, noise {u['sigma']}, loss {u['p']}</p>
<div class="card"><div class="verdict">{esc(vd['action'])}</div><ul>{"".join(f"<li>{esc(x)}</li>" for x in vd['reasons'])}</ul>
<p class="mut">Best case (LP oracle, perfect information): {hh} lower max-utilisation than ECMP.</p></div>
<h2>Gain over ECMP by telemetry quality</h2><div class="card">{"".join(svg)}<div class="legend">{legend}</div></div>
<h2>Details</h2><div class="card"><table><tr><th>telemetry</th><th>method</th><th>gain %</th><th>p99 regret</th><th>% intervals worse than ECMP</th></tr>{"".join(rows)}</table>
<p class="mut">Regret = max-util minus ECMP's; positive means worse than ECMP. "yours" is the telemetry you specified.</p></div>
<h2>Polling interval ({esc(o['polling_sweep']['method'])})</h2><div class="card"><table><tr><th>poll every</th><th>gain %</th><th>p99 regret</th></tr>{ps}</table></div>
<p class="mut">Offline estimate from a fluid-model simulation, not a deployment recommendation or a certificate for production. Intervals are 95% bootstrap over trace windows. Use shadow mode on real telemetry before any rollout.</p>
</main></body></html>"""


def main(a):
    topo, labels, builtin = load_network(a.network)
    trace = load_trace(a.traffic, topo, labels) if a.traffic else builtin
    tele = TelemetryCfg(a.polling, a.delay, a.noise, a.loss) if a.profile is None else PROFILES[a.profile]
    out = audit(topo, trace, tele, a.load, a.horizon, a.seeds, a.model)
    base = Path(a.out)
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix(".json").write_text(json.dumps(out, indent=1))
    base.with_suffix(".html").write_text(to_html(out), encoding="utf-8")
    print(out["verdict"]["action"], *out["verdict"]["reasons"], sep="\n  ")
    print(f"wrote {base.with_suffix('.html')}")
