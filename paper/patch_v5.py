"""One-off editing script: restructure paper.md into the TNSM-oriented v5 text. Idempotence is not guaranteed; run once on the v4 file."""
import re
from pathlib import Path

root = Path(__file__).resolve().parent
s = (root / "paper.md").read_text(encoding="utf-8").replace("\r\n", "\n")
new_results = (root / "new_results.md").read_text(encoding="utf-8").replace("\r\n", "\n")


def sub(a, b, count=1):
    global s
    assert a in s, "missing: " + a[:70]
    s = s.replace(a, b, count)


def cut(start, end, new):
    """Replace s[start:end) where start/end are marker strings (end exclusive)."""
    global s
    i, j = s.index(start), s.index(end)
    s = s[:i] + new + s[j:]


# ---- abstract
cut("## Abstract", "## 1 Introduction", """## Abstract

Which traffic-engineering controller should an operator run when its telemetry is stale or noisy? We compare ECMP, a reactive heuristic re-tuned for each telemetry condition, and a learned split-ratio policy trained only on small synthetic graphs. All three run under the same impairment model, in a fluid simulator and an emulated SDN, on 24 held-out Topology Zoo graphs and replayed SNDlib traces. The answer depends on the telemetry. With clean or moderately impaired telemetry, the re-tuned heuristic matches or beats the learned policy. With delay, noise and drops together, the learned policy is ahead. On Zoo graphs it gains +0.033 dScore (95% interval [+0.018, +0.049]) and loses on none of the 24 graphs. It is also ahead in all nine real-trace cells. Its mean p99 regret is lower in every profile (0.017 to 0.019 against 0.036 to 0.092), but this tail advantage is not uniform. It has the lower tail on only 9 to 13 of 24 graphs, and the gap comes from a few graphs where the heuristic does badly. In a delay and polling grid, the heuristic's gain falls as both grow, while the learned policy's stays flat. The effect is clear on GEANT and unresolved on Abilene, and no single delay flips the ranking. Ablations show that fluid-model features matter and message passing does not. A single tuning run can shift a comparison as much as the effects reported. Telemetry faults are modelled, not measured, so this is not a production validation.

""")

# ---- results 6.1 .. 6.2a replaced
cut("### 6.1 Held-out Topology Zoo graphs (24 graphs)", "### 6.2b Sensitivity to the load-scaling rule", new_results)

# ---- 6.2 oracle sentence is useful; keep in 6.2 text
sub("The real-trace results agree with the Zoo results.", "The oracle gain (for graphs with at most 25 nodes) is 47% to 58% on Abilene and 37% on Germany17, and the zero-shot learned policy captures roughly half of it. The real-trace results agree with the Zoo results.")

# ---- 6.4 overload
cut("### 6.4 Where it fails: overload", "### 6.5 Guard", """### 6.4 Where it fails: overload

Real-trace windows in which ECMP's own mean max-utilisation exceeds 2 are traffic surges far above the requested load, and they lie outside the training range (loads 0.5 to 1.1). We compared the methods on 30 Abilene windows at load 0.8 (Table 8). Of these, 27 are typical and 3 are overloaded.

**Table 8.** Typical and overloaded Abilene windows at load 0.8 (30 windows). Gain over ECMP in %: median and worst window.

| telemetry | window type | windows | heuristic gain (median, worst) | learned gain (median over seeds, worst) | learned + guard (median, worst) | learned worse than ECMP in |
|---|---|---|---|---|---|---|
| ideal | typical | 27 | 41.1, 8.6 | 36.5, 2.1 | 28.6, 0.0 | 0% of windows |
| ideal | overloaded | 3 | 49.0, 48.3 | 9.1, 4.6 | 19.9, 18.3 | 33% of windows |
| degraded | typical | 27 | 28.6, -21.5 | 35.3, 1.4 | 29.0, 0.1 | 0% of windows |
| degraded | overloaded | 3 | 42.5, 40.9 | 20.8, 6.1 | 26.8, 25.6 | 17% of windows |

In typical windows the heuristic is better under ideal telemetry, and the learned policy is better and never worse than ECMP under degraded telemetry, where the heuristic's worst window loses 21.5% against ECMP. In the three overloaded windows the heuristic gains 41% to 49% and the learned policy 9% to 21%, and the learned policy is worse than ECMP in one of the three. The guard lifts the learned policy to 20% to 27% in these windows. Tail statistics pooled over windows are therefore dominated by a few surges.

""")

# ---- 6.5 guard: tables captions, remove heuristic-original sentence
sub("Zoo graphs (10 graphs with n <= 25, `table_guard_audit.md`), headline guard (L2 only, eps = -0.06):\n\n| scenario",
    "Table 9 reports the headline guard (layer 2 only, eps = -0.06) on 10 Zoo graphs with at most 25 nodes.\n\n**Table 9.** Guard behaviour on 10 Zoo graphs. Fallback is the share of intervals replaced by ECMP; false rejection is the share of fallbacks where the proposal was beneficial; harm caught is the share of harmful proposals replaced; gain kept compares the guarded with the unguarded gain.\n\n| scenario")
sub("its value on real traces appears in overloaded windows (Section 6.4) and for the tuned heuristic. On the 24 test graphs",
    "its value on real traces appears in overloaded windows (Section 6.4). On the 24 test graphs")
sub("""It falls back often (71-80%) yet keeps 74-91% of the gain because fallback occurs mostly in low-gain intervals. For the
tuned heuristic under degraded telemetry the guard improves the gain by 2.3 points on real traces and lowers its p99 regret
from 0.198 to 0.074; the validation-chosen guard barely affects the learned policy on real traces (0.0-0.8 points).""",
    """It falls back often (71% to 80%) yet keeps 74% to 91% of the gain because fallback occurs mostly in low-gain intervals. On real traces the validation-chosen guard barely affects the learned policy (0.0 to 0.8 points of gain).""")
sub("**Guard against the re-tuned heuristic** (`table_v2_guard_stress.md`; 24 Zoo graphs, L2-only guard, eps = -0.06; mean dScore, p99 regret):\n",
    "Table 10 applies the same guard to the learned policy and to the re-tuned heuristic on all 24 Zoo graphs.\n\n**Table 10.** Guard applied to the learned policy and to the re-tuned heuristic (24 Zoo graphs, layer 2 only, eps = -0.06). Each cell gives mean dScore and mean per-graph p99 regret.\n")

# ---- 6.6
sub("""a
5% threshold cuts this to 8% and a 10% threshold to 2.4% with unchanged gain (3.1%) (`table_ratelimit.md`); the tuned
heuristic already updates about 2.5% of demands. On the real traces (`table_ratelimit_real.md`, degraded telemetry) the learned policy""",
    """a
5% threshold cuts this to 8% and a 10% threshold to 2.4% with unchanged gain (3.1%). On the real traces (degraded telemetry) the learned policy""")
sub("update threshold (`table_emu2.md`, controller with the fail-safe fixes, 28 runs). Simulator and emulator agree:", "update threshold (28 runs). Simulator and emulator agree:")
sub("Means over the four scenarios (emulated max-util / flow-mods per minute / control loop):\n\n| policy [telemetry]",
    "Table 11 gives means over the four scenarios.\n\n**Table 11.** Emulation results, means over four scenarios: emulated max-utilisation, flow-mods per minute and controller loop time.\n\n| policy [telemetry]")
sub("| original tuned heuristic [SNMP] | 0.798 | 33 | 9 |\n| original tuned heuristic [degraded] | 0.808 | 30 | 9 |\n| re-tuned heuristic [SNMP]", "| re-tuned heuristic [SNMP]")
cut("The re-tuned heuristic rows come from a re-run", "Decision latency", """The re-tuned heuristic runs agree with the simulator as well (8 runs, correlation 1.00, mean absolute difference 0.018). Under SNMP telemetry the re-tuned heuristic (0.772) is close to the learned policy (0.763), but it reprograms about 140 flows per minute against 270 to 332 for the learned policy, so the learned policy costs about twice the flow updates. Under degraded telemetry the learned policy is ahead (0.772 against 0.806, loss 0.0080 against 0.0095) in all four scenarios, by only 0.001 in one of them. Under SNMP the two are level on average; the learned policy is lower in three scenarios and higher in one. A 20% update threshold cuts the learned policy's updates 4.4-fold (332 to 76 per minute) at a cost of 0.018 in max-utilisation, leaving it ahead of the re-tuned heuristic under degraded telemetry (0.790 against 0.806). In the emulator the guard costs more than in simulation (0.794 against 0.772 max-utilisation, and packet loss 0.0143 against 0.0080 for the unguarded policy). Loop times were measured with the host otherwise idle. Capacities are scaled down and traffic is synthetic, so absolute rates do not transfer.

""")
sub("Decision latency (single CPU thread, p50 / p99 over 100 intervals, `table_latency.md`)", "Decision latency (single CPU thread, p50 and p99 over 100 intervals)")

# ---- 6.7 audit tool
cut("Run on four real networks with degraded telemetry", "The tool therefore reports intervals and ties", """We ran the audit tool on four real networks with degraded telemetry (polling every 5 intervals, delay 3, noise 0.2, loss 0.1; 6 trace windows), using the re-tuned heuristic. For Abilene it recommends the tuned heuristic (30.6%, 95% interval [27.2, 34.1]; best-case headroom 56.5%), which it cannot distinguish from the learned policy or its guarded version on six windows. For GEANT it recommends the learned policy (37.5% [23.1, 51.0]) and warns that the heuristic is worse than ECMP in 7% of intervals and that no method keeps p99 regret under 0.02. For Germany50 it recommends the learned policy (33.9% [30.7, 36.8]), not distinguishable from its guarded version. For Brain it recommends staying on ECMP, flagged provisional because six windows cannot rule out the rare surge window in which the learned policy gained 21%. With the single-run heuristic the Abilene and GEANT recommendations were the opposite. With six windows and close calls, the recommendation depends on the heuristic's tuning, which is a reason to run the tool on more windows. """)

# ---- 6.3 table captions / labels
sub("Held-out windows (last 30% of the trace), gain % (`experiments/final/table_ft.md`; seed-0 GNN):\n\n| Abilene @0.8 | tuned |",
    "Table 12 compares the zero-shot policy with models that use a network's own history, on held-out windows (the last 30% of the trace).\n\n**Table 12.** Held-out gain (%) on Abilene at load 0.8. The heuristic column is the single-run heuristic of Appendix A, not the re-tuned one.\n\n| Abilene @0.8 | single-run heuristic |")
sub("Ablations (24 Zoo graphs; `table_ablations.md`, one seed, and `table_gnn_vs_mlp.md`, four training seeds each for the full\nmodel, the per-link MLP and the no-fluid-feature model):",
    "Ablations (24 Zoo graphs; one seed for the history and age ablations, and four training seeds each for the full model, the per-link MLP and the no-fluid-feature model):")

# ---- scaling 6.2b placeholder filled later by apply_scaling(); marker kept
# ---- section 5 protocol seeds
sub("5 evaluation seeds, horizon 100, load 0.8;", "5 evaluation seeds per graph, horizon 100, load 0.8;")
s = re.sub(r"Deviations from the planned protocol \(4 instead of 10 training seeds, 5-12 instead of 20 evaluation\nseeds, single-seed ablations, post-hoc baselines and analyses\) are listed in `docs/eval_protocol.md`\.",
           "Evaluation uses 5 seeds per Zoo graph and 12 to 16 random windows per real-trace cell, fewer than the 20 seeds we planned. We train four policy seeds, not ten, and some ablations use one seed. Several baselines and analyses were added after the first results (the re-tuned heuristic, the delay grid and the window-level intervals), and a list of deviations from the original protocol is in the supplementary material.", s)

# ---- related work: DATE predecessor and Scholar sentence
sub(" and its predecessor DATE (IWQoS 2021)", "")
s = re.sub(r"In our literature search \(a few query variants plus full-text\s+reading of the papers above\) we found no paper that sweeps polling period, delay, noise, missing counters and install\s+latency together for centralised split-ratio TE; this is a search result, not proof of absence, and a Scholar check on\s+\"measurement staleness SDN TE\" should precede submission \(`docs/related_work\.md`\)\.",
           "In our literature search we found no paper that sweeps polling period, delay, noise, missing counters and install latency together for centralised split-ratio TE. This is a search result and not proof of absence.", s)

# ---- contributions: "open audit tool"
sub("(v) an open audit tool that applies\nthese measurements to a user's network.", "(v) an audit tool that applies\nthese measurements to a user's network.")

# ---- limitations
cut("## 8 Limitations and status of claims", "## 9 Reproducibility", """## 8 Limitations and status of claims

* The fluid model has no TCP, queueing or packet dynamics. The guard offers no formal guarantee. It is an empirically evaluated filter, and it misses most harmful proposals under degraded telemetry on Zoo graphs.
* Real traffic is scaled to a target load by factors from 0.02 to 23 (Section 3). Trace capacities for Germany17 and Germany50 are not meaningful in absolute terms, and some traces contain extreme outlier intervals. The conclusion is insensitive to the scaling rule (Section 6.2b), but the absolute numbers are not.
* Telemetry faults are modelled, not measured on a deployed network.
* We train four policy seeds. The history and age ablations and the DOTE-style and fine-tuned models use one seed. Evaluation uses 5 seeds per Zoo graph and 12 to 16 windows per real-trace cell.
* The heuristic's tuning budget is 80 trials per tuner seed on four synthetic graphs. Tuner-seed spread reaches about 0.02 dScore on Zoo graphs and about 10 points of gain in the delay grid, so single-run comparisons are fragile. We did not test tuning the heuristic on a network's own history, which the fine-tuned learned variants receive.
* Germany50, Brain and the history-based comparison (Table 12) use a single-run heuristic (Appendix A) and were not repeated with the re-tuned heuristic.
* The DOTE-style and fine-tuned models need per-network history, and only Abilene and GEANT have enough. Germany17 and Germany50 have one day of data, and their windows overlap.
* Out-of-range overload hurts the zero-shot policy (Section 6.4).
* We do not compare with Teal or other published systems. Teal optimises total flow or maximum link utilisation against LP, NCFlow and POP baselines with accurate demand inputs and treats staleness only as solver run time, so it is not a like-for-like baseline for telemetry impairments. Our DOTE-style model is a re-implementation of the idea, not the authors' code.
* We do not claim a general performance win, being first to show degradation under delay (LOGGIA), or production readiness.

""")

# ---- reproducibility + references
cut("## 9 Reproducibility and availability", "## References", """## 9 Reproducibility and availability

Code, configuration, trained policies and result tables are in the project repository ([repository URL to be inserted at submission]). The environment is Python 3.9 or later (tested with 3.13 on Windows 11 and in the Docker images), installed with `pip install -e ".[dev,rl]"` (torch CPU), and the 39 unit tests run with `python -m pytest -q`. The SNDlib matrices are public (Abilene, GEANT, Germany17, Germany50 and Brain) and are not redistributed; they are downloaded and prepared with the repository's `prepare` command. Topology Zoo graphs are bundled under their original terms. Each table has one script, and the artefact map in the repository README lists which script and result file produce each table. Seeds, window selection (random 100-interval windows chosen by integer seed), tuner seeds (100 to 104), the model-selection rule and the bootstrap procedures (percentile, 4000 resamples, fixed seed) are in the scripts and in Section 5. The heuristic is selected by its best tuning score on four synthetic graphs, and the learned policy uses a fixed training schedule with no selection on test data. Emulation (Docker, Mininet, Ryu) runs in real time and needs an otherwise idle machine. A clean-environment reproduction check (fresh clone, tests, data download and a table slice) is recorded in the repository.

""")

# ---- references: drop DATE, fix FlexDATE and SNDlib
s = re.sub(r"\n11\. M\. Ye, J\. Zhang, Z\. Guo, H\. J\. Chao\. DATE:.*?\n", "\n", s, flags=re.S)
s = s.replace("12. S. Orlowski", "11. S. Orlowski")
s = s.replace("(data source; verify citation details before release)", "(data source)")
s = s.replace("*IEEE/ACM Trans. Networking*, 2022 (online; issue 2023). doi:10.1109/TNET.2022.3217083.", "*IEEE/ACM Trans. Networking* 31(4), 2023. doi:10.1109/TNET.2022.3217083.")
s = s.replace("Entries below were checked against publisher, proceedings or arXiv pages on 2026-10-14.\n\n", "")

# ---- Appendix A
appendix = """## Appendix A: Single-run heuristic results

Our first comparisons used a heuristic tuned once per profile (40 trials, one run). We report them here because they differ from the re-tuned results in the main text, and the difference is itself a finding: a single tuning run can favour the learned policy by a wide margin.

**Table A1.** Zoo graphs with the single-run heuristic. Difference is heuristic minus learned (positive favours the learned policy), with a 95% bootstrap interval over graphs.

| telemetry | heuristic dScore | difference [95% CI] | learned W/T/L | heuristic p99 regret |
|---|---|---|---|---|
| ideal | -0.080 | -0.016 [-0.030, -0.005] | 1/10/13 | 0.024 |
| rich | -0.067 | -0.004 [-0.014, +0.004] | 6/12/6 | 0.033 |
| SNMP | -0.049 | +0.013 [+0.003, +0.023] | 9/12/3 | 0.048 |
| degraded | +0.014 | +0.075 [+0.036, +0.120] | 14/10/0 | 0.183 |

With the single-run heuristic the learned policy appeared ahead under SNMP telemetry and ahead by more than twice the margin under degraded telemetry. On real traces (Abilene, GEANT and Germany17, 12 windows per cell, 36 cells), the learned policy appeared ahead in 32 cells, the heuristic ahead in 2 and tied in 2. Recomputing the heuristic column with the re-tuned parameters on the same windows gives 12, 17 and 7 (Table 3). The single-run heuristic also fell below ECMP under degraded telemetry on Germany17 (3% to 4% worse) and in individual Abilene windows (by up to 51%).

Two networks were evaluated only with the single-run heuristic. On Germany50 (50 nodes, larger than any training graph) the learned policy gained 33% to 36% against 9% to 31% for the heuristic. On Brain the gain was about 0.3% for every method in 12 windows. The overload analysis and emulation in Sections 6.4 and 6.6 also give single-run heuristic results in the repository, and the qualitative conclusions match those in the main text.

"""
sub("## References", appendix + "## References")
s = s.replace("(a ", "(a ", 1)
(root / "paper.md").write_text(s, encoding="utf-8")
print("patched; words:", len(s.split()))
