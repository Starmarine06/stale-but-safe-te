# Evaluation protocol (frozen before final experiments)

Status: **frozen at commit of this file**. Any change after the first final run must be recorded in the
"Deviations" section at the bottom, with the reason.

## Claim under test
Learned graph-RL routers evaluated with perfect telemetry degrade under realistic telemetry; training with
impairments plus a runtime guard recovers most of the gain and bounds worst-case regret vs ECMP.

## Environment
- Fluid flow-level simulator (`src/stale_but_safe/env`), decision interval = 1 step, horizon 100 steps/episode.
- k = 4 candidate paths/demand (loop-free shortest simple paths by hop count).
- Capacities: degree rule `cap = 2.5*sqrt(deg_u*deg_v)/2` (floored at 0.5, 0.5 grid) for all graphs.
- Load = ECMP max-utilisation of the base matrix: {0.5, 0.8, 1.1}. Main results at 0.8.
- Traffic regimes: `gravity` (burst_sigma 0.25, hotspot_prob 0.02), `hotspot` (hotspot_prob 0.1),
  `shift` (pattern change at step 50), `flash` (flash crowd 20 steps at step 40).
- Failures: 1 random bidirectional link fails at step 40 (scenario `fail`).
- Install latency L = 0 in the main sweep; L in {0, 2} in the sensitivity sweep.

## Telemetry
Profiles (P poll period, d delay, sigma noise, p drop): ideal (1,0,0,0), rich (1,1,.03,0),
snmp (5,1,.05,.02), degraded (5,3,.2,.1). One-factor sweeps around `snmp`: d in {0,1,2,3,5}, P in {1,2,5,10},
sigma in {0,.05,.1,.2,.4}, p in {0,.05,.1,.2,.4}.

## Topologies (fixed in `configs/splits.yaml`, seed 12345)
- **Train**: synthetic BA/ER/geometric, n in [10, 20], seeds >= 1000 (never reused for tuning or test).
- **Validation** (hyper-parameters, early stopping, guard epsilon): 12 Topology-Zoo graphs, n <= 25.
- **Test** (never touched during development): `test_subset` = 24 stratified held-out Zoo graphs
  (n = 11..50, includes graphs larger than any training graph) + Abilene.
- Heuristic tuning topologies use seeds 9001-9004 (`eval/tune.py`).

## Methods
SPF, ECMP, Greedy, Tuned heuristic (EWMA/hysteresis, random search budget 40, **tuned separately for each
telemetry profile** on the tuning topologies under that profile), LP oracle (perfect info, upper bound),
GROM-style GNN-PPO baseline (simplified re-implementation, ideal-telemetry training, no history, no mask),
GNN-PPO ideal-trained, GNN-PPO telemetry-randomized (ours), ours + guard (epsilon sweep), ablations
(MLP encoder, no history, no age mask).

**Matched budget:** the agent gets the same number of hyper-parameter configurations (40) as the heuristic,
selected on validation graphs only; heuristic params are selected on its own tuning graphs under the
target profile.

## Metrics
Per interval: max link utilisation, utilisation variance, loss, mean path delay, churn.
Primary: `improvement_vs_ecmp = 1 - mean(max_util)/mean(ecmp_max_util)` and
**regret = max_util - ecmp_max_util** (positive = worse than ECMP) with mean, p95, p99, CVaR95.
Because loss dominates at load >= 0.8 we also report a composite
`score = max_util/ecmp_max_util + 5*loss` (lower is better).
Safety: fallback trigger rate, time-in-fallback. Operational: inference time vs n, churn.

## Statistics
- Evaluation traces: seeds 5000-5019 per (topology, scenario); paired across methods (identical traffic).
- Training seeds: as many as compute allows (target 10; **report the actual number**).
- Mean with 95% bootstrap CI (10 000 resamples over seeds x topologies); interquartile mean and
  performance profiles following *rliable*; Wilcoxon signed-rank (paired, per topology) with Holm correction.
- Never select among runs on test data. Test graphs are evaluated once per final configuration.

## Sim -> emulation
Same scenario in the simulator and in Mininet/Ryu (scaled-down capacities, real polling delay).
Report Pearson/Spearman correlation of max-util traces and rank agreement of methods.

## Known limitations (to appear in the paper)
Fluid model (no packet dynamics / TCP); empirical, not formal, safety bound; Mininet capacities scaled down;
synthetic traffic only unless a real TM dataset is added; Zoo graphs have no real capacities.

## Deviations
Everything below differs from the protocol above or was added after seeing results. Post-hoc analyses are exploratory.

| # | Planned | Actually done | Why / consequence |
|---|---|---|---|
| 1 | 10 training seeds | **4** per headline configuration (randomised-telemetry and ideal-telemetry); **1** per ablation, per DOTE-style model, per fine-tuned model | Compute. Seed SD of the learned policy is 0.001-0.002 (Zoo score), up to ~5 gain points on Abilene; single-seed ablation differences below ~0.004 are not interpretable |
| 2 | 20 evaluation seeds | Zoo: **5** per graph; real traces: **4** windows (first run), **10** independent windows (second run; 12 for Brain, loads 0.8 only) | Compute. Germany17/50 have only one day of data (288 intervals), so their windows overlap and are not independent |
| 3 | Full Zoo test split (141 graphs) | 24-graph `test_subset`; oracle only on graphs with n <= 25 (12 graphs) | LP cost; oracle numbers on the oracle subset must not be compared with 24-graph means |
| 4 | Synthetic traffic only unless a real dataset is added | **Real SNDlib traces added** (Abilene, GEANT, Germany17, Germany50, Brain), scaled per (network, load) so the median ECMP max-utilisation equals 0.6/0.8/1.0; Brain restricted to its 600 heaviest demands (90% of sampled traffic) | Load scaling means absolute load is not the original one; sensitivity to the scaling rule is NOT yet tested |
| 5 | Baselines: ECMP, tuned heuristic, LP oracle | **DOTE-style per-network MLP, static-LP routing and a per-network fine-tuned GNN added after seeing results** | Reviewer-style objection; DOTE-style trained on first 70% of each trace, tested on last 30%; Abilene and GEANT only (others too short) |
| 6 | Guard settings chosen on validation | Done (validation sweep over eps, tau, layers); the headline guard (L2-only, eps=-0.06) was chosen on validation graphs and evaluated once on test. Guard audit (false rejection, harm caught) is post-hoc and exploratory | |
| 7 | - | Overload-window analysis, loss-weight sensitivity, ablations, rate-limit sweep: post hoc | Exploratory |
| 8 | Tuned heuristic tuned per profile | 40-trial random search on four synthetic graphs per telemetry profile; **not** re-tuned for real traces | Real-trace results favour neither side by design, but the heuristic may be under-tuned for real traffic |
| 9 | Emulation: multiple scenarios | See `experiments/final/table_emu*.md` for the scenarios and seeds actually run | |
| 10 | Telemetry faults measured | **Modelled** (P, d, sigma, p, L); parameters from common SNMP practice, not from a deployed network | Main external-validity limit |
