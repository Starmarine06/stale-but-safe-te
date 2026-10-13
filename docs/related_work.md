# Related work notes (literature pass, Oct 2026)

**Status:** found by web search, **read at abstract/snippet level only**. Read each in full before citing.
This is not a systematic review.

## Closest to our claim — reposition against these
| Work | What it does | How we differ |
|---|---|---|
| *Towards Near-Real-Time Telemetry-Aware Routing with Neural Routing Algorithms* (LOGGIA), arXiv 2604.02927 | Models communication + inference delay in closed-loop neural routing; GNN predicts log-space link weights; pre-train then on-policy RL; reports neural baselines **fail once realistic delays are enforced**; best when fully local | Distributed per-router link-weight routing, packet-level. We study centralised SDN TE with split ratios over k paths, a **runtime safety guard with measured p99/CVaR regret vs ECMP**, **tuned-heuristic** baselines under the same impairments, polling/noise/drop impairments (not only delay) and Mininet/Ryu validation. Their finding supports our RQ1; we must not claim "first to show degradation under delay". |
| EnvelopeTC, arXiv 2605.09473 | Central controller bounds per-interval agent actions; after a staleness bound (500 ms) new reroutes are disabled (safe fallback) | Rule-based staleness cut-off (our L1 only). We add a model-based pre-action check (L2), a measured-regret monitor with backoff (L3), and quantify the gain-vs-worst-case trade-off. |
| GNN-guided OSPF reconfiguration with exact validator (Springer, 2026) | Reports fallback rate rising with snapshot delay (5.2%→11.2% at one noise level) | Learned component backed by exact validator for OSPF weight updates; different problem (metric updates, not split ratios). |
| Training Wheels / *Towards Safe Online RL in Computer Systems* (Mao et al., NeurIPS'19 workshop / MLSys'19) | Safe fallback policy takes over when state drifts to unsafe region; load-balancing demo | The conceptual basis of our guard. We cite it and do not claim the idea. |
| Safe online learning via smooth safety-structured composition, arXiv 2606.31320; universal policy wrapper, arXiv 2505.12354 | General fallback/Simplex-style wrappers | General RL, not telemetry-impaired TE. |

## GNN / DRL routing & TE (baseline literature)
* Teal (Microsoft WAN TE: GNN + multi-agent RL + ADMM); assumes accurate traffic matrices.
* CFR-RL (critical flow rerouting, ECMP default), DATE / FlexDATE (NYU), Almasan et al. *Deep RL meets GNNs*,
  ENERO, PPO-R, GTSR, GRL-RR, NeuroTE.
* Robust path selection in SD-WANs with DRL (arXiv 2212.11155): robustness to path *changes*, not measurement delay.
* Roracle (NYU): GNN trained on LP-optimal targets for future traffic matrices — **closest to our teacher
  distillation**; supervised, not DAgger/telemetry-impaired.
* DAgger under partial observability with a privileged oracle: imperfect-information games (arXiv 2012.12186);
  routing in similarity graphs (arXiv 1905.10987). We apply the idea to TE; we did not find it published for SDN TE.

## What we therefore claim (and don't)
* NOT novel: GNN-DRL TE; fallback to a safe policy; "neural routing degrades under delay" (LOGGIA).
* Our contribution: (i) a controlled impairment model (P, d, sigma, p, L) for **centralised SDN TE** that is shared by
  the simulator and a Ryu controller; (ii) the **heuristic-under-the-same-impairments** comparison; (iii) a
  **three-layer guard with measured worst-case regret** and its Pareto trade-off; (iv) privileged-teacher DAgger
  with randomised telemetry; (v) sim-to-Mininet validation. Whether (iv) beats a tuned heuristic is an empirical
  question — see results.

## Reading list (to verify before submission)
LOGGIA 2604.02927; EnvelopeTC 2605.09473; Training Wheels (MLSys'19); Roracle; Teal (SIGCOMM'23); CFR-RL; DATE;
Almasan et al. 2022; rliable (Agarwal et al., NeurIPS'21) for statistics.

## Full-text verification pass (agent-assisted; spot-check before submission)
Read in full text: LOGGIA, EnvelopeTC, DOTE, CFR-RL, Training Wheels, RedTE. Teal: grep only. **Not verified:**
FlexDATE/DATE, Roracle, FIGRET (abstract/snippet level).

* LOGGIA (Boltres et al., TMLR 2026, arXiv 2604.02927): centralised *and* distributed modes, B4/GEANT/synthetic, ns-3,
  single-path link-weight routing, models inference + propagation delay only; reports neural baselines no longer beat
  static shortest-path routing under delay. **Partial pre-emption** of "ranking flips": single axis (delay), RL vs
  default shortest-path, no tuned heuristic, no guard.
* EnvelopeTC (Jia et al., IEEE JCC 2026, arXiv 2605.09473): hierarchical SDN, bounds on edge agents, staleness bound
  disables reroutes, rollback/audit; no ranking claim; no SNDlib/Zoo.
* DOTE (Perry et al., NSDI 2023): direct optimisation of split ratios from the last 12 demand matrices; GEANT/Abilene/
  Microsoft traces; tests multiplicative demand noise up to +-35% only. **Training-method antecedent of ours; cite as such.**
* Teal (Xu et al., SIGCOMM 2023, arXiv 2210.13763): delay = solver latency only.
* CFR-RL (Zhang et al., IEEE JSAC 2020, arXiv 2004.11986): no staleness analysis; RL beats rule heuristics.
* RedTE (Gui et al., SIGCOMM 2024): centralised control-loop latency; moves decisions to routers.
* Training Wheels (Mao et al., NeurIPS 2019): safe-fallback precedent.
* Searches did not surface a paper that varies polling period, noise, missing counters and install latency together for
  centralised split-ratio TE (search-based, not exhaustive).

**Consequences for claims:** do not say "first to study stale telemetry in TE". Contribution = (i) multi-axis impairment
sweep, (ii) *tuned-heuristic* comparator re-tuned per telemetry profile, (iii) a ranking-flip analysis on real topologies and
traces, (iv) guard measured by tail regret / false rejection (idea from Training Wheels), (v) audit tool.

## Second full-text pass (agent-assisted)
FlexDATE (IEEE/ACM ToN 31(4), 2023; DATE = IWQoS 2021, DATE paper itself unopened), Roracle (ICNP 2023) and FIGRET (SIGCOMM 2024,
arXiv 2405.04932) read in full: none studies delayed/noisy/sampled/missing telemetry as a variable, none compares a tuned reactive
heuristic under impairment. Roracle is adjacent ("routing computed on the current TM goes stale; learned lookahead resists",
datasets include Abilene, GEANT, Brain). Closest 2023-2026 staleness papers (abstract level only): arXiv 2605.16187 (online TE,
control-loop latency), Teal, arXiv 2410.16846 (safe load balancing, two loops). Still to do: Scholar check, open the DATE paper.
