# Related work notes (literature pass, Oct 2026)

**Status:** found by web search; the detailed comparison below was checked against the full texts (see its own status line).
This is not a systematic review.

## Closest to our claim — reposition against these
| Work | What it does | How we differ |
|---|---|---|
| *Towards Near-Real-Time Telemetry-Aware Routing with Neural Routing Algorithms* (LOGGIA), arXiv 2604.02927 | Models communication + inference delay in closed-loop neural routing; GNN predicts log-space link weights; pre-train then on-policy RL; reports neural baselines **fail once realistic delays are enforced**; best when fully local | Distributed per-router link-weight routing, packet-level. We study centralised SDN TE with split ratios over k paths, a **runtime safety guard with measured p99/CVaR regret vs ECMP**, **tuned-heuristic** baselines under the same impairments, polling/noise/drop impairments (not only delay) and Mininet/Ryu validation. Their finding supports our RQ1; we must not claim "first to show degradation under delay". |
| EnvelopeTC, arXiv 2605.09473 | Central controller bounds per-interval agent actions; after a staleness bound (500 ms) new reroutes are disabled (safe fallback) | Rule-based staleness cut-off (our L1 only). We add a model-based pre-action check (L2), a measured-regret monitor with backoff (L3), and quantify the gain-vs-worst-case trade-off. |
| GNN-guided OSPF reconfiguration with exact validator (Springer, 2026) | Reports fallback rate rising with snapshot delay (5.2%→11.2% at one noise level) | Learned component backed by exact validator for OSPF weight updates; different problem (metric updates, not split ratios). |
| Training Wheels / *Towards Safe Online RL in Computer Systems* (Mao et al., NeurIPS 2019 workshop paper) | Safe fallback policy takes over when state drifts to unsafe region; load-balancing demo | The conceptual basis of our guard. We cite it and do not claim the idea. |
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
LOGGIA 2604.02927; EnvelopeTC 2605.09473; Training Wheels (NeurIPS 2019); Roracle; Teal (SIGCOMM'23); CFR-RL; DATE;
Almasan et al. 2022; rliable (Agarwal et al., NeurIPS'21) for statistics.

## Full-text verification pass (agent-assisted; spot-check before submission)
Read in full text: LOGGIA, EnvelopeTC, DOTE, CFR-RL, Training Wheels, RedTE. Teal: full text read 2026-10-14 (pdftotext of arXiv 2210.13763). **Not verified:**
FlexDATE/DATE, Roracle, FIGRET (abstract/snippet level).

* LOGGIA (Boltres et al., TMLR 2026, arXiv 2604.02927): centralised *and* distributed modes, B4/GEANT/synthetic, ns-3,
  single-path link-weight routing, models inference + propagation delay only; reports neural baselines no longer beat
  static shortest-path routing under delay. **Partial pre-emption** of "ranking flips": single axis (delay), RL vs
  default shortest-path, no tuned heuristic, no guard.
* EnvelopeTC (Jia et al., IEEE JCC 2026, arXiv 2605.09473): hierarchical SDN, bounds on edge agents, staleness bound
  disables reroutes, rollback/audit; no ranking claim; no SNDlib/Zoo.
* DOTE (Perry et al., NSDI 2023): direct optimisation of split ratios from the last 12 demand matrices; GEANT/Abilene/
  Microsoft traces; tests multiplicative demand noise up to +-35% only. **Training-method antecedent of ours; cite as such.**
* Teal (Xu et al., SIGCOMM 2023, arXiv 2210.13763), full text: staleness = solver run time only (an allocation persists until the
  scheme finishes computing; LP-all at 585 s reuses stale routes from several 5-min intervals, satisfied demand 87.2% vs Teal 90.6%).
  Objectives: total flow / MLU on a Microsoft-style WAN (ASN) trace; baselines LP-all, LP-top (demand pinning), NCFlow, POP; no ECMP,
  no tuned reactive heuristic, no telemetry impairment (noise, drops, polling, delay of counters), and the authors state Teal "overfits" the
  topology and degrades on out-of-distribution demands. Not a baseline for our setting (different objective and action space, trained
  per topology, assumes accurate TMs); we do not compare against it and say so.
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
control-loop latency), Teal, arXiv 2410.16846 (safe load balancing, two loops). Still to do: Scholar check; the DATE paper (IWQoS 2021) could not be located by web search on 2026-10-14 (search returned an unrelated DATE: Distributed Adaptive TE, JSAC 2007) and remains unopened.

## Search log, 2026-10-14 (browser session)
* Google Scholar returned a bot-check page ("unusual traffic"); not bypassed, so no Scholar check was done. Semantic Scholar API rate-limited.
* arXiv API, 13 keyword queries combining traffic engineering with stale/delayed/outdated/telemetry/noisy and learning/RL/GNN/ECMP/DOTE:
  no paper found that sweeps telemetry impairments with a re-tuned heuristic and ECMP. Only new adjacent hit: arXiv 2212.11155
  (Robust Path Selection in SD-WANs using DRL: path-update overhead and route computation, not telemetry quality). Keyword search on arXiv is
  weak (exact-match), so this is absence of evidence, not proof of novelty; a manual Scholar/ACM/IEEE Xplore pass is still needed.


## Detailed comparison (2026-10-22; PDFs read with pdftotext, entries below checked against the text)
Read in full or in the relevant sections: LOGGIA (arXiv 2604.02927v3), Roracle (ICNP 2023 PDF), DOTE (earlier pass), FIGRET (arXiv 2405.04932, baselines and robustness sections), EnvelopeTC (arXiv 2605.09473v2), Safe Load Balancing (arXiv 2410.16846).

| | This study | LOGGIA [7] | Roracle [6] | DOTE [1] | FIGRET [2] | EnvelopeTC [8] | Safe LB [11] |
|---|---|---|---|---|---|---|---|
| Control problem | Centralised split ratios over k = 4 candidate paths, one decision per telemetry interval | Link weights (single-path), centralised or per-router, sub-second | Lookahead routing: predict a routing for future TMs from the last TMs | Split ratios over paths from recent demand matrices | Split ratios with burst-aware robustness | Controller envelopes + edge agents (metering, queueing, reroute) | DRL load balancing with a safety shield |
| Observation model | Utilisation and demand counters through a channel with polling period, delay, noise, dropped entries, install latency | INT-based snapshots with propagation, communication and inference delay; no noise or loss of counters | Measured TMs; staleness = routing computation delay and traffic drift | Demand matrices (ideal); multiplicative demand noise tested up to +-35% | Demand matrices (ideal) | sFlow/INT-style telemetry, 500 ms staleness bound on envelopes | Network statistics (ideal) |
| Impairment varied as an experimental variable | Five axes, one at a time and jointly, plus delay-by-polling grid | One axis (inference delay scaling; deployment mode fixes communication delay) | Age of the routing (how long a stale routing is applied) | Demand noise | None for telemetry | None (staleness bound is a rule) | None |
| Baselines | ECMP, heuristic re-tuned per profile (5 tuner seeds x 80 trials), static LP, DOTE-style, LP oracle | OSPF/EIGRP/RIP shortest-path variants (static), neural routers | Optimal routing on the current TM and other TE schemes | LP-based TE schemes | DOTE, desensitisation-based and LP-based TE | Static ECMP, centralised TE (500 ms loop), threshold reroute, learned agents | Non-RL load balancers |
| Evaluation level | Fluid simulator on 24 Zoo graphs + SNDlib traces; Mininet/Ryu with UDP and TCP | ns-3 packet-level, TCP/UDP traffic, synthetic and real topologies | Trace-driven (Abilene, CERNET, GEANT, BRAIN) | Trace-driven (GEANT, Abilene, production traces) | Trace-driven (WAN and data-centre) | 1,024-host software SDN testbed | Packet-based simulator |
| Objective | Max link utilisation (+ loss in the score) | Goodput | Max link utilisation | Max link utilisation | Max link utilisation | Utilisation, FCT, SLA | End-to-end delay under capacity safety |
| Main finding | Ranking depends on telemetry quality and on baseline tuning; learned lead not established at packet level | Neural baselines no longer beat shortest-path under delay; fully local deployment works best | Stale routing degrades by more than 20% MLU; lookahead resists it | Direct optimisation matches or beats LP-based TE | Fine-grained robustness reduces burst-case MLU | Edge learning inside envelopes beats centralised TE | CBF keeps DRL inside the safe region |

**Exact novelty claim.** Not new: that delayed or stale inputs degrade learned routing (LOGGIA, Roracle); learned split-ratio TE trained by direct optimisation (DOTE, FIGRET); safe fallback (Training Wheels, EnvelopeTC, Safe LB). New is the combination of (a) a five-axis telemetry channel shared by the simulator and a real controller reading OpenFlow counters, (b) a heuristic baseline re-tuned per impairment profile with tuning-seed variability reported and ranking stability measured, (c) zero-shot evaluation on 24 held-out Zoo graphs and replayed SNDlib traces with a packet-level TCP check, and (d) the resulting findings: the ranking changes with telemetry quality, a poorly tuned baseline can flip a conclusion, and the simulated lead is not established at packet level. The contribution is an evaluation, not an algorithm.

**What would weaken it.** A paper that varies several telemetry axes against a per-condition re-tuned heuristic for centralised TE; or a re-run of LOGGIA inside our setting showing the same ranking dependence. LOGGIA's code is public; running it with our heuristic and ECMP as baselines is listed as future work, not done.
