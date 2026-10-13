# Telemetry Quality Determines When Learned Traffic Engineering Outperforms Reactive Heuristics

*Draft v2. All numbers come from scripts in `experiments/`; raw results are in `experiments/final/`. Items marked
**[PENDING]** depend on experiments still running when this draft was written. Protocol deviations are tabulated in
`docs/eval_protocol.md`; claims and their status are in Section 8.*

## Abstract

Learned traffic-engineering (TE) policies are usually evaluated with accurate, timely link counters, whereas
operational controllers see counters that are polled every few minutes, delayed, noisy and sometimes missing. We ask
how telemetry quality changes which method is worth deploying. We define a shared impairment model (polling period,
delay, noise, missing counters, rule-install latency) used by a fluid simulator and an SDN controller (Ryu with Open
vSwitch in Mininet), and compare ECMP, a tuned reactive heuristic (re-tuned for each telemetry profile), static and
per-network learned (DOTE-style) baselines, an LP oracle, and a policy trained only on small synthetic graphs. On 24
held-out Topology Zoo graphs the tuned heuristic is better with clean telemetry, while the learned policy is better
with degraded telemetry (21-23 of 24 graphs over four training seeds), where the heuristic's average score falls to
ECMP level. On replayed SNDlib traces from four networks with traffic-engineering headroom, scaled to controlled loads,
the learned policy is never behind the heuristic under non-clean telemetry across ten independent trace windows per
cell, and the heuristic falls below ECMP on two of four networks. The picture is conditional: a DOTE-style model
trained on a network's own history beats the zero-shot policy (48-49% versus 22-25% gain on Abilene), fine-tuning on
history closes most of that gap, an ablation shows that model-based fluid features rather than graph message passing
carry the result, and in overloaded trace windows the heuristic beats the learned policy. A runtime ECMP-fallback guard
reduces tail regret in some regimes (link failures) and little in others; a hysteresis rate limiter removes most flow
updates at no measured loss of gain. Telemetry faults are modelled, not measured, and traffic is scaled to a target
load, so the results do not establish performance in a production network.

## 1 Introduction

Telemetry quality is a missing dimension in the evaluation of learned traffic engineering. Learned TE policies are
typically trained and tested with perfect inputs; operational controllers poll counters at fixed periods and act on
stale, noisy and incomplete views. Heuristics tuned to the latest counters can chase stale information, and a learned
policy trained on clean inputs may misbehave. Prior work shows that neural routers degrade under realistic delay
(Section 2), so we do not claim that observation. We ask a narrower, operational question:

> For a given network and telemetry quality, which of ECMP, a tuned reactive heuristic, or a learned policy should an
> operator run, and what does each cost in tail risk and in controller activity?

We contribute (i) an impairment model shared by simulator and controller and a like-for-like comparison against a
heuristic re-tuned for every telemetry profile; (ii) evidence on held-out Topology Zoo graphs and replayed real traces
that the ranking depends on telemetry quality, with the conditions under which it does not (clean telemetry,
overload, per-network training data); (iii) an ablation and baseline study that identifies what carries the result;
(iv) an empirical evaluation of an ECMP-fallback guard and of a rate limiter; and (v) an open audit tool that applies
these measurements to a user's network. Negative results (LP-teacher distillation, PPO fine-tuning) are reported.

## 2 Related work

*Learned TE.* DOTE (Perry et al., NSDI 2023) trains a network on recent demand matrices to output split ratios over
candidate paths by directly minimising maximum link utilisation; our training method is of this family, and we add a
differentiable fluid model with packet-loss term and training under impaired observations. Teal (Xu et al., SIGCOMM
2023), CFR-RL (Zhang et al., IEEE JSAC 2020), FlexDATE and Roracle target scalability or rerouting; they assume accurate
inputs. DOTE tests demand noise up to +-35% only; Teal's delay is solver latency. *Delay and telemetry.* LOGGIA
(Boltres et al., TMLR 2026, arXiv 2604.02927) models communication and inference delay for neural link-weight routing
in a packet-level simulator and reports that neural baselines stop beating static shortest-path routing under delay; it
compares with default shortest-path protocols, studies one impairment axis, and has no safety mechanism. This is the
closest prior result and partially anticipates our central claim. EnvelopeTC (Jia et al., IEEE JCC 2026) bounds agent
actions and disables reroutes after a staleness limit in a hierarchical data-centre setting. RedTE (Gui et al., SIGCOMM
2024) addresses centralised control-loop latency by moving decisions to routers. *Safe fallback.* Training Wheels (Mao
et al., NeurIPS 2019) falls back to a simple policy in unsafe states; our guard applies this idea. In a literature search
we found no study that varies polling period, noise, missing counters and install latency together for centralised
split-ratio TE and compares learned policies against heuristics re-tuned per telemetry profile; the search was not
exhaustive. FlexDATE, Roracle and FIGRET were read at abstract level only (`docs/related_work.md`).

## 3 Setting

**Network and traffic.** A topology with link capacities and k = 4 candidate paths per demand; the policy outputs split
ratios each interval. Traffic is synthetic (gravity base, diurnal cycle, bursts, hotspots, shifts, flash crowds) or a
replayed SNDlib trace (Abilene, 5-min, 48,096 matrices; GEANT, 15-min, 11,460; Germany17 and Germany50, 5-min, 288 each;
Brain, 1-min, 9,723, restricted to its 600 heaviest demands = 90% of sampled traffic). Each (network, load) pair uses one
scale factor so that the median ECMP max-link-utilisation over the trace equals 0.6, 0.8 or 1.0. This is *real-trace
evaluation under controlled load scaling*: temporal and spatial structure are real, absolute load is not.

**Telemetry impairment.** The controller observes utilisation and demands through a channel with polling period P
(intervals), delay d, multiplicative noise sigma and per-counter drop probability p; installing rules takes L
intervals. Profiles: ideal (1,0,0,0), rich (1,1,0.03,0), SNMP-like (5,1,0.05,0.02), degraded (5,3,0.2,0.1). The same
channel is applied to real OpenFlow counters in the emulated controller.

**Metrics.** We report separately: gain (reduction of mean max-link-utilisation vs ECMP), fluid packet loss, tail
regret (p99 of per-interval max-util minus ECMP's; positive = worse than ECMP) and flow-update rate. The composite score
(max-util ratio + 5 x loss, paired difference to ECMP, "dScore") is a convenience; conclusions do not depend on the
weight (Section 6.1). **Unit of analysis:** for Zoo results, the graph (paired across methods; seeds and training seeds
are averaged within a graph and reported as spread); for real traces, the (network, load) cell with independent trace
windows inside it; windows within one day-long trace overlap and are not treated as independent.

## 4 Methods

**Baselines.** *ECMP.* *Tuned heuristic:* EWMA-smoothed bottleneck soft-min over candidate paths with hysteresis, damped
updates and age-based shrinkage of stale readings; five parameters tuned per telemetry profile by a 40-trial random
search on four synthetic graphs disjoint from training and test graphs (it therefore knows the telemetry profile, a
favourable baseline). *Static LP:* LP-optimal splits for the mean traffic matrix of the training part of a trace, no
telemetry. *DOTE-style:* an MLP on the last three impaired demand estimates, trained per network on the first 70% of its
trace (ideal-telemetry and degraded-telemetry variants) with our differentiable loss; evaluated on the last 30%.
*LP oracle:* perfect information, graphs with n <= 25 only; a reference bound, not a deployable method.

**Learned policy.** A GNN on the link graph with path pooling. Inputs: impaired utilisation and demand history (H = 3),
telemetry age, and model-based fluid features (ECMP-predicted utilisation, utilisation predicted for the current
configuration, per-path what-if bottleneck). Training is direct optimisation through a differentiable fluid model: the
policy sees the impaired observation, the loss uses the true next demand, `smoothmax(util)/ecmp_mu + 5 loss + 0.1 churn`.
Training graphs are 40 synthetic BA/ER/geometric graphs with 10-20 nodes; no Zoo or SNDlib graph is used. Variants:
randomised-telemetry training ("ours") and ideal-only training; four seeds each. *Fine-tuning:* the pre-trained model is
trained for 6 iterations (~3-5 minutes on CPU) on rollouts over the first 70% of a network's trace with randomised
telemetry and evaluated on the last 30%.

**Guard.** L1: stale or missing telemetry, malformed action, deadline. L2: fluid-model check of the proposal against ECMP
on the *estimated* state, rejecting it if predicted max-util exceeds ECMP's by more than a margin eps. L3: runtime
monitor with backoff. The headline guard is L2 only with eps = -0.06, chosen on validation graphs. **Property claimed:**
none formal. The guard is an empirically evaluated runtime filter; we report its fallback rate, false-rejection rate,
the share of harmful proposals caught and the regret before and after (Section 6.5).

**Rate limiter.** A demand's installed split is kept unless the proposal moves it by more than a threshold (half the L1
distance); the number of updates per interval is the flow/group modifications a controller would send.

**Audit tool.** `stale-but-safe audit` takes a topology, optional traffic and telemetry parameters and reports best-case
headroom (LP oracle where n <= 25), gain with a 95% bootstrap interval over trace windows and tail regret for each
method under the user's telemetry, a polling sweep, and one of three outcomes: stay on ECMP, use a named method, or
insufficient evidence (too few windows, interval straddling the 5% threshold, or gain concentrated in one window). It
flags telemetry values outside the evaluated range. Its output is an offline estimate, not a deployment recommendation.

## 5 Protocol

Topology Zoo: 263 graphs split into train/validation/test (`configs/splits.yaml`); we report the 24-graph `test_subset`,
5 evaluation seeds, horizon 100, load 0.8; paired bootstrap 95% CI, IQM and Holm-corrected Wilcoxon over graphs.
Real traces: four networks with headroom plus Brain, loads {0.6, 0.8, 1.0}, four profiles; two independent evaluation
runs (4 windows, then 10 fresh windows per cell, 12 for Brain at load 0.8). Tuning and model selection never use test
graphs or windows. Deviations from the planned protocol (4 instead of 10 training seeds, 5-12 instead of 20 evaluation
seeds, single-seed ablations, post-hoc baselines and analyses) are listed in `docs/eval_protocol.md`.

## 6 Results

### 6.1 Held-out Topology Zoo graphs (24 graphs)

| telemetry | tuned dScore | learned dScore (4 seeds, mean +- SD) | learned beats tuned on (per seed) | p99 regret tuned / learned |
|---|---|---|---|---|
| ideal | -0.080 | -0.065 +- 0.002 | 7-9 of 24 | 0.024 / 0.018 |
| rich | -0.067 | -0.063 +- 0.002 | 14-16 of 24 | 0.033 / 0.018 |
| SNMP | -0.049 | -0.062 +- 0.001 | 16-18 of 24 | 0.048 / 0.017 |
| degraded | +0.014 | -0.061 +- 0.001 | 21-23 of 24 | 0.183 / 0.019 |

(Seed 0, Holm-adjusted Wilcoxon over graphs, tuned vs learned: p = 0.013, 0.84, 0.078, 0.0012. Bootstrap 95% CI of the
tuned heuristic under degraded telemetry: [-0.023, +0.057], i.e. not distinguishable from ECMP.) The ranking is
insensitive to the packet-loss weight: with weight 0 (pure max-utilisation) the tuned heuristic still wins at ideal
telemetry (-0.056 vs -0.043) and the learned policy wins at degraded telemetry (-0.040 vs -0.006); see
`experiments/final/table_lossweight.md`. Under link failures, demand shifts, flash crowds and hotspots (12 graphs,
SNMP/degraded telemetry) the learned policy is ahead in every case (`experiments/final/stress_*.json`).

### 6.2 Real traces

Tables: `experiments/final/table_real.md` (first run, 4 windows) and `table_realw.md` (10 fresh windows). Over the four
networks with headroom (Abilene, GEANT, Germany17, Germany50 in the first run; first three in the second), per
(network, load) cell, the learned policy (mean of four training seeds) versus the tuned heuristic:

* first run (12 cells per profile): rich, SNMP-like and degraded telemetry, learned ahead in 12/12 each; clean telemetry
  7 learned, 4 tuned, 1 tie;
* second run (9 cells per profile): rich 8 ahead + 1 tie, SNMP 8 + 1, degraded 9 ahead; clean telemetry 4 learned, 3 tuned,
  2 ties. "Ahead" means more than one point of gain.
* The heuristic falls below ECMP under degraded telemetry on Germany17 (-3% to -4%) and, in individual Abilene windows,
  by up to -51% (Section 6.4); on GEANT it falls to about 0% at degraded telemetry on the held-out windows, and on
  Germany50 it does not collapse.
* Germany50 (50 nodes, larger than any training graph): learned 33-36% versus 9-31%.
* Brain: gain is about 0.3% for every method in the 12 windows of the second run; in one window of the first run the
  learned policy gained 20.9%, so benefits are confined to rare windows.
* The oracle gain (n <= 25) is 47-58% on Abilene and 37% on Germany17; the zero-shot policy captures roughly half.

### 6.3 What carries the result: baselines and ablations

Held-out windows (last 30% of the trace), gain % (`experiments/final/table_ft.md`; seed-0 GNN):

| Abilene @0.8 | tuned | static LP | DOTE-style (degraded-trained) | GNN zero-shot | GNN fine-tuned |
|---|---|---|---|---|---|
| ideal | 33.6 | 34.5 | 49.1 | 23.7 | 46.2 |
| SNMP | 27.7 | 34.5 | 49.1 | 22.4 | 41.3 |
| degraded | 23.5 | 34.5 | 48.9 | 22.5 | 39.3 |

On GEANT the DOTE-style model reaches about 18% in every profile, the zero-shot policy 15-18%, the fine-tuned policy 16-19%
and static LP -38.6% (traffic is not stationary). Hence: (a) with per-network history a DOTE-style model beats the
zero-shot GNN, and fine-tuning recovers most of the difference on Abilene but little on GEANT; (b) robustness to
telemetry is a property of any policy that does not rely on fresh counters, not specific to our GNN; (c) what the
zero-shot policy offers is no need for per-network training data.
Ablations (24 Zoo graphs, one seed each; `table_ablations.md`): removing the fluid-model features costs +0.008 (ideal) and
+0.013 (degraded) dScore and raises p99 regret from 0.019 to 0.030; removing history costs +0.003 at ideal telemetry only;
removing the age input is ~0; replacing the GNN with a per-link MLP is slightly better (-0.005, -0.004). The model-based
features, not message passing, carry the result.
Training with randomised telemetry versus ideal-only: the mean differences are within seed noise (~0.002), but the
randomised model has a lower p99 regret under degraded telemetry (0.019 vs 0.027 over four seeds).

### 6.4 Where it fails: overload

Real-trace windows in which ECMP's own mean max-utilisation exceeds 2 (traffic surges far above the requested load) lie
outside the training range (loads 0.5-1.1). On Abilene at load 0.8 over 30 windows: in 27 typical windows the learned
policy has median gain 36.5% and is never worse than ECMP, while the tuned heuristic has median 31.5% (worst window
-50.8% under degraded telemetry); in 3 overloaded windows the tuned heuristic gains 45-49% and the learned policy 9-21%
(worse than ECMP in one of three; worst case +3 to +7 in absolute max-util). The guard lifts the learned policy to 20-27%
in those windows (`table_overload.md`). Tail statistics pooled over windows are therefore dominated by a few surges.

### 6.5 Guard

Definitions (per interval, using the true demand, simulator only): *beneficial* proposal = true max-util below 0.99 x ECMP's,
*harmful* = above 1.01 x; false rejection = beneficial proposals among fallbacks; harm caught = harmful proposals replaced.
Zoo graphs (10 graphs with n <= 25, `table_guard_audit.md`), headline guard (L2 only, eps = -0.06):

| scenario | fallback | false rejection | harm caught | gain kept | p99 regret raw -> guarded |
|---|---|---|---|---|---|
| ideal | 71% | 12% | 43% | 83% | +0.050 -> +0.019 |
| degraded | 72% | 14% | 22% | 76% | +0.065 -> +0.057 |
| link failure, degraded | 80% | 13% | 90% | 91% | +0.148 -> +0.021 |
| demand shift, degraded | 73% | 18% | 53% | 74% | +0.084 -> +0.058 |

On real Abilene and GEANT traces (load 0.8, three typical windows each) the guard fires in only 0-2% of intervals because
the learned policy's proposals are essentially never harmful there (0.0% of accepted intervals harmful; raw p99 regret
already <= 0); its value on real traces appears in overloaded windows (Section 6.4) and for the tuned heuristic. On the 24 test graphs the per-graph p99 regret falls from 0.019 to 0.008
at degraded telemetry (average of per-graph p99), while the pooled p99 above falls only from +0.065 to +0.057: the guard cuts
the typical tail but misses harmful proposals concentrated in a few graphs. The guard is most effective under link failures.
It falls back often (71-80%) yet keeps 74-91% of the gain because fallback occurs mostly in low-gain intervals. For the
tuned heuristic under degraded telemetry the guard improves the gain by 2.3 points on real traces and lowers its p99 regret
from 0.198 to 0.074; the validation-chosen guard barely affects the learned policy on real traces (0.0-0.8 points).

### 6.6 Operational cost and emulation

Without a rate limiter the learned policy updates 91% of demands every interval in simulation (Zoo, ideal telemetry); a
5% threshold cuts this to 8% and a 10% threshold to 2.4% with unchanged gain (3.1%) (`table_ratelimit.md`); the tuned
heuristic already updates about 2.5% of demands. **[PENDING: rate-limit sweep on Abilene and GEANT.]**

Emulation: Mininet with Open vSwitch and a Ryu controller reading real OpenFlow counters, 24 flows, four scenarios
(three traffic seeds on a 10-node graph, one 12-node random graph), six policy/telemetry runs each plus a run with a 20%
update threshold (`table_emu2.md`, controller with the fail-safe fixes, 28 runs). Simulator and emulator agree: across the 24
runs at the default 5% threshold the correlation between emulated and simulated mean max-utilisation is 0.99, the mean
absolute difference 0.017 and the bias +0.010; per-run time-series correlation is 0.83-0.98; the policy ranking agrees
(Kendall tau 0.47-0.73 per scenario). Means over the four scenarios (emulated max-util / flow-mods per minute / control loop):

| policy [telemetry] | max-util | flow-mods/min | loop ms |
|---|---|---|---|
| ECMP | 0.866 | 0 | 9 |
| tuned [SNMP] | 0.798 | 33 | 9 |
| tuned [degraded] | 0.808 | 30 | 9 |
| learned [SNMP] | 0.763 | 270 | 12 |
| learned [degraded] | 0.772 | 332 | 14 |
| learned [degraded], threshold 0.2 | 0.790 | 76 | 12 |
| learned + guard (L2, eps -0.06) [degraded] | 0.794 | 258 | 12 |

The learned policy gives lower max-utilisation than the tuned heuristic but reprograms about ten times more flows; a 20%
update threshold cuts updates 4.4-fold (332 to 76 per minute) at a cost of 0.018 in max-util, leaving it ahead of the tuned
heuristic under degraded telemetry (0.790 vs 0.808). In the emulator the guard costs more than in simulation (0.794 vs 0.772
max-util, and packet loss 0.0143 vs 0.0080 against the unguarded policy). Loop times were measured with the host otherwise
idle; capacities are scaled down and traffic is synthetic, so absolute rates do not transfer.

### 6.7 Audit tool

On real Abilene with degraded telemetry the tool reports 47-58% best-case headroom, flags the tuned heuristic as fragile and
recommends the learned policy. For Brain and for windows with surges, the new interval logic returns "insufficient evidence"
or "stay on ECMP" depending on the interval; the earlier version, which used three windows, was window-sensitive.

## 7 Negative results

LP-teacher distillation (DAgger) produced a weak student (6.4% gain vs 15.0% for the tuned heuristic on training graphs); PPO
fine-tuning did not change it; direct optimisation through the fluid model fixed this. Randomised-telemetry training did not
improve mean performance over ideal-only training. A per-link MLP did as well as the GNN. Static LP routing is excellent on
a stationary network (Abilene) and disastrous on a non-stationary one (GEANT).

## 8 Limitations and status of claims

* Fluid model: no TCP, queueing or packet dynamics. The guard offers no formal guarantee; it is an empirically evaluated filter
  and misses most harmful proposals under degraded telemetry on Zoo graphs.
* Real traffic but scaled to a target load; sensitivity to the scaling rule is **not tested**. Capacities are the networks'
  published module sizes where available.
* Telemetry faults are modelled, not measured on a deployed network.
* Four training seeds (not ten); ablations, DOTE-style and fine-tuned models have one seed; evaluation seeds 4-12 (not 20).
* DOTE-style and fine-tuned models need per-network history; only Abilene and GEANT have enough.
* Germany17 and Germany50 have one day of data; their windows overlap.
* Out-of-range overload hurts the zero-shot policy (6.4).
* No comparison with Teal or other published systems; DOTE-style is our re-implementation of the idea, not the authors' code.
* Not claimed: a general performance win; being first to show degradation under delay (LOGGIA); production readiness.

## 9 Reproducibility

`pip install -e ".[dev]"`, `python -m pytest -q`; real data from SNDlib (public), prepared by
`python -m stale_but_safe.env.real prepare`; one script per table in `experiments/`; emulation in `docs/howto.md`.
