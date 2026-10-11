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
