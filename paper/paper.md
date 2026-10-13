# When does learned traffic engineering beat a tuned heuristic? Telemetry quality decides: a benchmark, a safety guard and an audit tool

*Draft. All numbers are produced by the scripts in `experiments/`; results files are in `experiments/final/`.
Status of claims is stated explicitly in Section 8. Items marked **TBD** are not finished.*

## Abstract

Learned traffic-engineering (TE) policies are usually evaluated with perfect, instantaneous link counters.
Real controllers see counters that are polled every few minutes, delayed, noisy and sometimes missing. We
study how this changes which method wins. We build a controlled telemetry-impairment model (polling period,
delay, noise, missing counters, rule-install latency) shared by a fluid simulator and an SDN controller
(Ryu + Open vSwitch in Mininet), and compare ECMP, a per-profile tuned heuristic, an LP oracle and a graph-neural-network
(GNN) split-ratio policy trained by direct optimisation through a differentiable fluid model. The policy is
trained only on small synthetic graphs. On 24 held-out Topology Zoo graphs and on real SNDlib traffic traces
(Abilene, GEANT, Germany17, Germany50, Brain), the ranking depends on telemetry quality: with clean telemetry a
tuned heuristic is competitive and often better; with degraded telemetry it can fall to ECMP level or worse
(for example 4.0% *worse* than ECMP on Abilene at load 1.0, worst-case regret +1.02), while the learned policy keeps
its gain (about 29% there). In the four real networks with TE headroom, the learned policy is ahead of the tuned
heuristic in all 36 cells with rich, SNMP-like or degraded telemetry, and ahead in 7 of 12 with clean telemetry.
A validation-chosen pre-action safety guard roughly halves the learned policy's worst-case regret at a cost of
about 18% of its gain, and rescues the tuned heuristic. Finally, we package the measurements as an audit tool that
tells an operator, for a given topology, traffic and telemetry quality, whether TE is worth deploying at all (on
one real network, Brain, benefits are confined to rare traffic windows) and which method to trust.

## 1 Introduction

Most learned-TE papers assume the controller knows the current traffic matrix and link loads. Operational
networks do not: SNMP counters are typically polled every few minutes, flow telemetry is sampled, and both
arrive late and with gaps. A heuristic that reacts to the latest counters can then chase stale information, and a
learned policy trained on clean inputs can misbehave. Prior work shows neural routers degrade under realistic
delay (LOGGIA, arXiv 2604.02927, for distributed link-weight routing); we do not claim that observation. Our
questions are different and operational:

* **RQ1.** In centralised SDN TE, how does the ranking between a tuned heuristic and a learned policy change with
  telemetry quality, on real topologies and real traffic?
* **RQ2.** Can a policy trained only on synthetic graphs under randomised telemetry transfer, and does the
  randomisation matter?
* **RQ3.** How much of the worst-case risk can a runtime guard remove, and at what cost in gain?
* **RQ4.** Do simulator conclusions hold in an emulated SDN?
* **RQ5.** Can the measurements be turned into a decision aid for an operator?

Contributions: (i) a telemetry-impairment model shared by simulator and controller; (ii) a like-for-like
comparison of tuned heuristics and a learned policy under the *same* impairments on real traces; (iii) a
three-layer guard with measured tail regret and a validation-based selection protocol; (iv) a sim-to-emulation
check; (v) an open audit tool. Negative results (LP-teacher distillation, PPO fine-tuning) are reported.

## 2 Related work

*Learned TE.* Teal (GNN + multi-agent RL + ADMM), DOTE (direct optimisation of the TE objective from historical
matrices), CFR-RL, DATE, Roracle (GNN trained on LP-optimal targets). These assume accurate traffic inputs.
*Delay and telemetry.* LOGGIA models communication and inference delay for neural routing and reports that
neural baselines fail under realistic delay; EnvelopeTC bounds agent actions and disables reroutes after a
staleness bound. *Safe fallback.* "Training Wheels" (Mao et al.) hands control to a safe policy when the state
becomes unsafe; our guard builds on that idea and adds a model-based pre-action check and a measured-regret
monitor. We read most related papers at abstract level only; they must be verified before submission
(`docs/related_work.md`).

## 3 Setting

**Network and traffic.** A topology with link capacities and `k = 4` candidate paths per demand. Traffic is either
synthetic (gravity base, diurnal cycle, bursts, hotspots, shifts, flash crowds) or a *real trace* replayed from
SNDlib. Real traces are scaled by one constant per (network, load) so that the median ECMP max-link-utilisation
over the trace equals the requested load (0.6, 0.8 or 1.0); temporal and spatial structure remain real, absolute
load does not. The policy outputs split ratios over the candidate paths each interval.

**Telemetry impairment.** The controller observes link utilisation and demands through a channel with polling
period `P` (intervals), delay `d`, multiplicative noise `sigma` and per-counter drop probability `p`; rule
installation takes `L` intervals. Four profiles: *ideal* (1,0,0,0), *rich* (1,1,0.03,0), *SNMP* (5,1,0.05,0.02) and
*degraded* (5,3,0.2,0.1). The same channel is applied to real counters in the emulated controller.

**Metrics.** Gain = reduction of mean max-link-utilisation vs ECMP; score = max-util ratio vs ECMP plus 5 x
fluid packet loss, reported as the paired difference to ECMP (dScore, lower is better); tail risk = p99 of
(max-util minus ECMP's max-util) per interval ("regret", positive = worse than ECMP).

## 4 Methods

**Baselines.** ECMP; a tuned heuristic (EWMA-smoothed bottleneck soft-min over candidate paths with hysteresis, damped updates and
age-based shrinkage of stale readings; its five parameters are tuned separately for each telemetry profile by a
40-trial random search on four synthetic graphs disjoint from the training and test graphs, which gives it the
information that telemetry quality varies, a favourable baseline); an LP
oracle with the true next demand (graphs with n <= 25 only).

**Learned policy.** A GNN on the link graph with path pooling. Inputs are impaired utilisation and demand history
(H = 3), telemetry age, and model-based fluid features (ECMP-predicted utilisation, utilisation predicted for the
current configuration, per-path what-if bottleneck). Training is *direct optimisation through a differentiable
fluid model*: the policy sees only the impaired observation, while the loss uses the true next demand,
`smoothmax(util)/ecmp_mu + 5 fluid_loss + 0.1 churn` (DOTE-style). Training graphs are 40 synthetic BA/ER/geometric
graphs with 10-20 nodes, never Topology Zoo or SNDlib graphs. Two variants: *ours* (telemetry randomised per
episode) and *ideal* (clean telemetry only). Four training seeds each.

**Guard.** L1: stale/missing telemetry, malformed action, deadline; L2: fluid-model check of the proposal against
ECMP on the estimated state with margin eps; L3: runtime monitor with exponential backoff. Settings are chosen on
validation graphs only; the headline guard is L2 only with eps = -0.06.

**Audit tool.** `stale-but-safe audit` takes a topology (SNDlib name, Topology Zoo graph or GraphML), optional
traffic (CSV or trace) and telemetry parameters, and reports best-case headroom (LP oracle), gain and tail regret of
each method under the user's telemetry, a polling-interval sweep and a rule-based recommendation, as a
self-contained HTML report.

## 5 Protocol

Topology Zoo: 263 graphs split into train/validation/test by a fixed seed (`configs/splits.yaml`); we report a
24-graph test subset (`test_subset`), 5 evaluation seeds, horizon 100, load 0.8. Statistics are computed over
topologies (paired): bootstrap 95% CI, IQM, Holm-corrected Wilcoxon. Real traces: five networks, three loads, four
profiles, four evaluation seeds (each replays a different random window); the learned result is the mean over four
training seeds. Guard and tuned-heuristic parameters never use test graphs. Deviations from the pre-registered
protocol (fewer training and evaluation seeds than planned) are listed in `docs/eval_protocol.md`.

## 6 Results

### 6.1 Held-out Topology Zoo graphs (24 graphs)

| telemetry | tuned dScore | learned dScore (4 seeds, mean +- SD) | learned beats tuned on (per seed) | p99 regret tuned / learned |
|---|---|---|---|---|
| ideal | -0.080 | -0.065 +- 0.002 | 7-9 of 24 | 0.024 / 0.018 |
| rich | -0.067 | -0.063 +- 0.002 | 14-16 of 24 | 0.033 / 0.018 |
| SNMP | -0.049 | -0.062 +- 0.001 | 16-18 of 24 | 0.048 / 0.017 |
| degraded | +0.014 | -0.061 +- 0.001 | 21-23 of 24 | 0.183 / 0.019 |

(Seed 0: tuned vs learned Holm-adjusted Wilcoxon p = 0.013, 0.84, 0.078, 0.0012 for ideal/rich/SNMP/degraded.)
With clean telemetry the tuned heuristic is better; with degraded telemetry it is no better than ECMP on average
(95% CI [-0.023, +0.057]) and has a tail ten times worse. The seed-to-seed spread of the learned policy (0.001-0.002)
is much smaller than the effects above. Under four stress scenarios (link failure, demand shift, flash crowd,
hotspots; 12 graphs, SNMP/degraded telemetry) the learned policy is ahead of the tuned heuristic in every case
(`experiments/final/stress_*.json`).

### 6.2 Real traces

Table `experiments/final/table_real.md` (60 cells). Summary over the four networks with TE headroom (Abilene,
GEANT, Germany17, Germany50; 12 network/load cells per telemetry profile):

* rich, SNMP, degraded telemetry: learned ahead of tuned in 12/12, 12/12, 12/12 cells (more than one point of gain);
* clean telemetry: learned 7, tuned 4, tie 1. The tuned heuristic wins on GEANT and Germany17 at loads 0.8 and 1.0
  by 1.5-4.1 points.
* Largest tuned failures under degraded telemetry: Abilene at load 1.0, -4.0% gain (worse than ECMP, p99 regret
  +1.02) against +29.4% for the learned policy; Abilene at 0.8, 3.5% against 26.4% with clean telemetry; Germany17,
  -3%. The failure is network dependent: on Germany50 and (mildly) GEANT the tuned heuristic does not collapse.
* Germany50 (50 nodes, larger than every training graph): learned 33-36% gain against 9-31% for tuned, in all 12 cells.
* Brain (161 nodes, one-minute traces, 600 heaviest demands = 90% of traffic): TE headroom is concentrated in rare
  windows. In 11 of 12 trace windows tested the learned policy gains 0.0-0.3%; in one window it gains 20.9%, which
  produces the 3-5% averages in the table. A single 100-interval window therefore under- or over-states the benefit
  here, and the audit tool's verdict for such networks should be read with that variability in mind.
* The learned policy's gain varies by about +-5 points across training seeds on Abilene (+-1 on the larger graphs).
  The oracle gain (n <= 25 only) is 47% on Abilene and 37% on Germany17: the learned policy captures roughly half
  of the available headroom.

### 6.3 Guard

Validation-chosen L2-only guard (eps = -0.06), evaluated once on the 24 test graphs, degraded telemetry:
tuned +0.014 -> -0.034 (p99 regret 0.183 -> 0.030, Holm p = 0.005); learned -0.060 -> -0.049 (keeps 82% of the
gain; p99 regret 0.019 -> 0.008, p = 0.0003). Guarded learned still beats guarded tuned. The guard falls back to
ECMP in about 83% of intervals and costs about 0.011 of score even with clean telemetry. The earlier default
(eps = 0, all layers) was more conservative and halved the learned gain (`experiments/final/table_all.md`). On the real traces (matched seeds, seed-0 model) the same guard costs the learned policy only 0.0-0.8
points of gain on average per network (Abilene 0.2, GEANT 0.3, Germany17 0.8, Germany50 0.0, Brain 0.7) and cuts the
p99 regret on GEANT from +0.026 to +0.002; for the tuned heuristic under degraded telemetry it improves the gain by
2.3 points and cuts the p99 regret from 0.198 to 0.074 (no effect with clean telemetry). On Brain the guard falls
back 95-100% of the time because a 6% predicted improvement is rare, at a cost of about one point (it misses most
of the single high-gain window).

### 6.4 Ablations — **TBD**

Models without history (H = 1), without telemetry-age input, without the GNN (per-link MLP) and without model-based
features are trained (one seed) and being evaluated at ideal and degraded telemetry. Randomised-telemetry training
versus ideal-only: the mean gap is within seed noise (about 0.002), but randomised training has lower seed variance and a
lower p99 regret under degraded telemetry (0.019 vs 0.027 over four seeds).

### 6.5 Emulation

Seven runs of one scenario (10-node synthetic graph, 24 flows, one seed; capacities scaled down) in Mininet +
Open vSwitch with a Ryu controller reading real OpenFlow counters. Per-run time-series correlation between emulated and simulated
max-utilisation is 0.90-0.98; the mean difference in mean max-util is 0.011 (learned under degraded telemetry:
0.880 emulated vs 0.889 simulated; tuned 0.894 vs 0.914). Costs not visible in simulation: the learned policy
reprograms flows about ten times more often (265-408 vs 25-27 flow-mods/min) and its controller loop takes 26-32 ms
vs 16-18 ms. This is a validation of the simulator, not an independent result: one scenario, one seed.

### 6.6 Audit tool

On real Abilene with SNMP-like-to-degraded telemetry the tool reports 47-57% best-case headroom, flags the tuned
heuristic as fragile (worse than ECMP in 15% of intervals) and recommends the learned policy; on Brain (n > 25, so no exact optimum) it
reports that no tested method gains more than 0.1% in its three windows and recommends staying on ECMP; this verdict is
window-sensitive (see 6.2).

## 7 Negative results

LP-teacher distillation (DAgger) produced a weak student (6.4% gain vs 15.0% for the tuned heuristic on training
graphs); PPO fine-tuning did not change it; direct optimisation through the fluid model fixed this. Randomised
telemetry training did not improve mean performance over ideal-only training.

## 8 Limitations and status of claims

* Fluid model: no TCP, queueing or packet effects; the guard's guarantee is empirical, not formal.
* Real *traffic* but scaled to a target load; capacities are the networks' published ones where available (Abilene,
  GEANT, Germany: module capacities).
* Telemetry faults are *modelled*, not measured on a real network. Parameter values reflect common SNMP polling practice.
* Emulation: one scenario, one seed, scaled capacities; flow-mod rates reflect Open vSwitch on one machine.
* Learned policy trained once per seed with 4 seeds, not the protocol's 10; 4-5 evaluation seeds, not 20.
* The learned policy gets about half of oracle headroom; it is not a near-optimal solver. No comparison against
  published systems (DOTE, Teal) was run.
* "Learned beats tuned" holds under degraded telemetry; with clean telemetry the tuned heuristic is competitive.
* Not claimed: a general performance win; first demonstration of neural routing degrading under delay (LOGGIA).

## 9 Reproducibility

Code, configs and results: repository root. `pip install -e ".[dev]"`, `python -m pytest -q`, `docs/howto.md`
(emulation), `experiments/` (one script per table). Real data: SNDlib (public), prepared by
`python -m stale_but_safe.env.real prepare`.
