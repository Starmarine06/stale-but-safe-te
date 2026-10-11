# Research & Build Plan

## Title

**Delay-Robust and Safety-Guarded Graph Reinforcement Learning for Congestion-Aware Traffic Engineering in Software-Defined Networks**

Short name for the repo/talks: **Stale-but-Safe** (`stale-but-safe-te`)

Fallback titles, depending on what the results show:
- If the agent clearly beats tuned heuristics: *"…: Closing the Telemetry Gap Between Simulated and Deployable Learned Routing"*
- If it mostly doesn't: *"Do Graph-RL Routers Survive Real Telemetry? A Benchmark Against Tuned Heuristics in SDN"* (a benchmark / negative-result framing is still publishable if the evaluation is rigorous)

---

## 1. Novelty check (what the search found)

I ran targeted searches on each candidate angle. This is **not a systematic review**; do your own pass on Google Scholar / IEEE Xplore / arXiv before writing the paper (suggested queries in Section 14).

| Angle | What exists | Verdict |
|---|---|---|
| GNN + DRL routing that generalizes to unseen topologies and survives link/node failures | Crowded. Examples: Almasan et al., *Deep RL meets GNNs* (tested on 180 + 232 unseen topologies); *Intelligent routing optimization for SDN based on PPO and GNN* (PPO-R, handles node failures, unseen topologies); *Towards Robust Routing… Graph Transformers and DRL in SDN* (GTSR, topology changes); GRL-RR for LEO constellations; ENERO; CFR-RL | **Not novel by itself.** This is the baseline literature you build on, not your contribution. |
| Robustness to **noisy / delayed / partial telemetry** | Noise-robust training exists for traffic signal control (perturbing the state during training). In routing I found noisy-delay measurement handled with bandits + network tomography (Subito), but **no GNN-DRL TE paper that systematically studies stale, delayed, noisy or missing SDN telemetry** | **Most promising open gap.** Not proven empty; verify. |
| **Safe fallback** for learned systems | General idea exists: *Towards Safe Online RL in Computer Systems* (a guaranteed-safe fallback policy so the RL agent never does worse than it, shown on load balancing). I did not find it applied to GNN-DRL traffic engineering with measured worst-case regret vs ECMP | **Concept is not novel; the application + evaluation can be.** Cite the prior work, don't claim the idea. |
| **Honest benchmark vs tuned heuristics** | Done in other domains: RLScale-Bench (a calibrated rule-based autoscaler beat six DRL algorithms on cost); a GAT-RL paper for RWA found only <2% throughput gain over the best heuristic; a vehicle-routing study where nearest-feasible beat learned policies. Many SDN papers report big wins over **shortest-path/ECMP only**, i.e. weak baselines | **Open in SDN TE.** Strong baselines + seeds + distribution shift is itself a contribution. |

### Positioning

The three open pieces combine into one coherent story. **The claim:**

> Learned graph-RL routers are usually evaluated with perfect, instantaneous link state. Under realistic SDN telemetry (polling interval, delay, noise, loss), they degrade, and may lose to tuned heuristics. Training with telemetry impairments and wrapping the policy in a runtime safety guard recovers most of the gain while bounding worst-case regret versus ECMP, including zero-shot on unseen real topologies.

Everything in the project must serve this claim. If an experiment doesn't, cut it.

---

## 2. Research questions and hypotheses

| # | Research question | Hypothesis (falsifiable) |
|---|---|---|
| RQ1 | How much does a GNN-DRL TE policy trained on perfect state degrade under realistic telemetry impairments, compared with tuned heuristics under the same impairments? | H1: Degradation is large (e.g. loses most of its gain over ECMP) at multi-second polling/delay; heuristics degrade less. *(Might be false; report either way.)* |
| RQ2 | Does telemetry-randomized training (plus observation history) restore robustness? | H2: Yes, it recovers a substantial fraction of the lost gain without hurting the clean-telemetry case. |
| RQ3 | Does a runtime safety guard bound worst-case behaviour relative to ECMP, and what does it cost in average gain? | H3: The guard cuts p99 regret vs ECMP by a large factor for a small loss in mean improvement; the cost is tunable (Pareto curve). |
| RQ4 | Do conclusions hold zero-shot on unseen real topologies, and from simulation to Mininet/Ryu? | H4: Ranking of methods is preserved on held-out topologies; the sim→emulation gap is quantified and shrinks with domain randomization. |

**Pivot rule:** if after Phase 2 the LP oracle shows little headroom over ECMP, or the agent can't beat the tuned heuristic even with perfect telemetry, switch to the benchmark framing (RQ1 + RQ4 become the paper) rather than forcing a result.

---

## 3. Contributions (draft for the paper)

1. A **telemetry impairment model** for SDN TE (polling period, observation delay, noise, missing links) and a Gym environment implementing it, validated against Mininet/Ryu.
2. A **telemetry-aware GNN-PPO router** (history-stacked graph observations, randomized impairments at training time).
3. A **three-layer safety guard** with a tunable gain-vs-worst-case-regret trade-off.
4. A **rigorous benchmark** vs SPF, ECMP, tuned load-aware heuristics, an LP oracle and a re-implemented published GNN-DRL baseline, over multiple seeds, topologies and traffic regimes.
5. **Open-source code** with one-command reproduction.

---

## 4. Problem formulation

**Network:** graph G = (V, E), link capacities c_e, delays d_e. Demand matrix D_t (gravity, bursty, diurnal, real traces).

**Action (per decision interval Δ):** for each demand (s, t), split ratios over k precomputed candidate paths (k = 3–4), discretized (e.g. steps of 0.25). Fixed-size per-demand action keeps the policy topology-agnostic (the trick used by GROM-style methods).

**True state s_t:** link utilizations u_e(t), queue/loss, demands, link up/down.

**Observed state o_t (the contribution):**

```
o_t = M( s_{t-d}, noise σ, drop prob p )   with polling period P, delay d (in intervals), install latency L
```

- P: how often stats are polled (e.g. 1, 2, 5, 10 s)
- d: observation age at decision time
- σ: multiplicative noise on utilization (and on demand estimates)
- p: probability a link's counter is missing/stale this round
- L: rule-install latency before an action takes effect

The agent sees a short **history** of observations (H = 3–5 steps, or a GRU) with an **age/validity mask** per link so it can learn to discount stale data.

**Reward:** `r_t = −(α·max_util + β·mean_delay + γ·loss + δ·util_variance) − λ·flow_churn`. Start with max-util only; add terms one at a time and ablate. The churn term discourages route flapping.

**Policy:** GNN encoder (start GraphSAGE/GCN, compare GAT) → per-demand head → split logits; PPO. Compare against an MLP-on-flat-state ablation (to prove the GNN matters).

---

## 5. Safety guard design

Three layers; each is independently ablated.

1. **Hard rules (always on):** if stats are missing/older than a threshold, a link is down and routes are invalid, inference exceeds its deadline, or any split is malformed → use ECMP for this interval, log the event.
2. **Pre-action check (soft):** evaluate the proposed split vs ECMP with the fluid model on the *estimated* state plus an uncertainty margin. If predicted max-util > ECMP × (1 + ε), reject and use ECMP. **ε is the trade-off knob.**
3. **Runtime monitor:** track measured max-util vs the model-estimated ECMP counterfactual over a window W. If the policy is worse by more than τ for k consecutive intervals, **latch fallback** with exponential backoff before re-enabling.

**Be honest about guarantees:** this gives an *empirical* worst-case bound (p99 / CVaR regret vs ECMP), not a formal proof. Formal guarantees only hold under stated model assumptions; say so in the paper. Cite the prior "guaranteed-safe fallback" work as the conceptual basis.

---

## 6. System architecture (build target)

```
Traffic gen (iperf3 / D-ITG / replay) ──► Mininet topology (OVS, TCLink-limited bandwidth)
                                                │ OpenFlow 1.3
                                                ▼
                                       Ryu controller app
   ┌──────────────────┬────────────────────┬───────────────────┬──────────────────┐
   │ Stats poller     │ Observation builder│ Policy runner     │ Safety guard     │
   │ (port stats,     │ (graph + history + │ (TorchScript/ONNX │ (3 layers →      │
   │ period P)        │ age mask)          │ GNN-PPO)          │ ECMP fallback)   │
   └──────────────────┴────────────────────┴─────────┬─────────┴──────────────────┘
                                                      ▼
                          Flow installer (select groups w/ weighted buckets) + hysteresis
                                                      │
                                  Metrics logger → Prometheus/CSV → Grafana dashboard
```

Key engineering decisions:
- **Weighted split** via OpenFlow **select groups**; per-flow pinning if reordering is a problem. Add **hysteresis / min-hold time** to avoid flapping.
- Export the trained policy to **TorchScript/ONNX** so the controller doesn't need the training stack.
- **Scale link capacities down** (e.g. 10–100 Mbps) so Mininet isn't CPU-bound; note this limitation in the paper.
- Real **Ryu polling artifacts** give you genuine telemetry delay in emulation, which is a real advantage over pure simulation.

---

## 7. Experimental design

### Topologies
- **Train:** synthetic distribution (Barabási–Albert, Erdős–Rényi, random geometric; 10–50 nodes), with curriculum from small to large.
- **Validate:** a few Topology Zoo graphs.
- **Test (never touched during development):** held-out Topology Zoo graphs, including larger-than-train sizes.

### Traffic
Gravity model, hotspot/bursty, diurnal, elephant/mice mix, and **real traffic-matrix traces** (Abilene and GÉANT TM datasets are commonly used; confirm availability and licenses). Include light-load regimes where learning shouldn't help.

### Telemetry impairment sweeps
One-factor-at-a-time sweeps plus 3–4 named realistic profiles (avoids a combinatorial explosion):

| Profile | P | d | σ | p |
|---|---|---|---|---|
| Ideal (what most papers assume) | Δ | 0 | 0 | 0 |
| SNMP-like | 5–10 s | 1–2 | low | low |
| Telemetry-rich (sFlow/INT-like) | 1 s | 0–1 | low | 0 |
| Degraded | 5 s | 2–5 | high | 0.1 |

### Failures and shifts
Random link/node failures mid-episode; abrupt traffic-pattern shift; flash crowd.

### Methods compared
| Method | Role |
|---|---|
| SPF | trivial baseline |
| ECMP | the reference for "regret" |
| Greedy load-aware | simple heuristic |
| **Tuned** heuristic (EWMA smoothing, hysteresis, tuned thresholds; **tuned with the same budget as the agent**, under the same impaired telemetry) | the baseline reviewers will care about |
| LP min-max-util oracle (perfect info) | upper bound / headroom check |
| Re-implemented published GNN-DRL (simplified GROM / PPO-R style) | state-of-the-art-ish baseline |
| GNN-PPO trained on ideal telemetry | shows the problem (RQ1) |
| GNN-PPO telemetry-randomized (ours) | RQ2 |
| Ours + guard (ε sweep) | RQ3 |
| MLP-PPO, no-history, no-mask ablations | what actually matters |

### Metrics
- **Quality:** max link utilization, util variance, mean / p95 / p99 delay, throughput, loss
- **Safety:** regret vs ECMP (mean, p95, p99, CVaR), fallback trigger rate, time-to-recover
- **Operational:** stats→rules latency, inference time vs #nodes, flow-rule changes/min, reorder/flap rate
- **Generalization:** zero-shot vs trained-on-target gap, performance vs topology size
- **Sim→emulation gap:** same scenarios in sim and Mininet; compare metric correlation and rank agreement

### Statistical protocol
- ≥10 seeds for the agent (training seeds) and many evaluation traces; report mean with bootstrap CIs; use performance profiles / interquartile mean (the *rliable* approach), not just means.
- Paired comparisons on identical traffic traces; Wilcoxon signed-rank for pairwise tests.
- Matched tuning budgets for the agent and the heuristic; fix hyperparameter search before looking at test topologies.
- Pre-register the evaluation protocol in `docs/eval_protocol.md` **before** running final experiments.

---

## 8. Phased plan (≈14–16 weeks; scale to your deadline)

**Suggested roles (3 people):** A = environment + RL training, B = Ryu/Mininet integration + traffic, C = baselines + evaluation + paper/docs. Everyone reviews each other's PRs.

### Phase 0: Foundations (Weeks 1–2)
- [ ] Repo, CI (lint + unit tests), Docker image pinning Ubuntu 22.04, Python, Mininet, Ryu, PyTorch, PyG.
- [ ] Prove the toolchain: static shortest-path Ryu app on a 6-node Mininet topology with iperf3. **Do this before any ML**; Ryu/eventlet/Python version issues eat days.
- [ ] Do the literature pass (Section 14); lock the one-sentence claim.
- [ ] Write `docs/eval_protocol.md` draft.

### Phase 1: Environment + baselines (Weeks 2–4)
- [ ] Flow-level Gym env with k-shortest candidate paths, fluid delay model, loss when util > 1.
- [ ] **Telemetry impairment wrapper** (P, d, σ, p, L) as a first-class component with unit tests.
- [ ] Traffic generators + real-trace loader.
- [ ] All non-learning baselines incl. LP oracle and the tuned heuristic.
- [ ] Validate env vs Mininet on one topology (same traffic → similar utilization trends).
- **Gate 1:** LP oracle shows meaningful headroom over ECMP in some regimes (e.g. ≥10–15% lower max-util at medium/high load). If not, adjust traffic regimes or switch to the benchmark framing.

### Phase 2: Learning agent on ideal telemetry (Weeks 4–6)
- [ ] GNN-PPO on a single topology, ideal telemetry. Debug reward scaling, observation normalization, action masking.
- [ ] Ablate encoder (MLP vs GCN/SAGE/GAT), k, reward terms.
- **Gate 2:** agent beats ECMP **and the tuned heuristic** on the training topology under ideal telemetry. If it can't beat the tuned heuristic, stop and fix (reward, warm-start by imitating the oracle, smaller action space) or pivot to the benchmark framing.

### Phase 3: The core contribution (Weeks 6–9)
- [ ] Evaluate the ideal-trained agent and all baselines under impairment sweeps (RQ1). This result alone is informative.
- [ ] Add history + age mask + telemetry-randomized training (RQ2); compare.
- [ ] Implement the three-layer guard; sweep ε and produce the gain-vs-regret Pareto curve (RQ3).
- **Gate 3:** at least one clear, reproducible finding (positive or negative) with CIs.

### Phase 4: Generalization (Weeks 8–11, overlaps)
- [ ] Train on the synthetic distribution; zero-shot test on held-out Zoo topologies and larger sizes.
- [ ] Failure and traffic-shift scenarios.
- [ ] Compare against the re-implemented published GNN-DRL baseline.

### Phase 5: SDN validation (Weeks 9–12)
- [ ] Ryu app with stats poller, observation builder, policy runner, guard, select-group installer, hysteresis.
- [ ] Rerun key scenarios in Mininet with *real* Ryu polling delay; quantify the sim→emulation gap (RQ4).
- [ ] 30–60 min soak test; chaos tests (kill links, stall stats, restart controller, inference timeout).

### Phase 6: Usability + release (Weeks 12–13)
- [ ] `docker compose up` → topology + controller + traffic + dashboard; `make reproduce-fig3` style targets per paper figure.
- [ ] CLI: `stale-but-safe demo --topology abilene --mode shadow|active`.
- [ ] Docs: quickstart, add-a-topology, plug-in-a-policy, limitations. Tag v0.1.
- [ ] **One outside user** runs it from the README alone; fix what breaks.

### Phase 7: Paper (parallel from Week 4; final push Weeks 13–16)
- [ ] Draft Related Work, Formulation, Methodology during the build; fill Results as experiments land.
- [ ] Every figure generated by a script; no hand-made plots.
- [ ] Internal review + advisor feedback before submission.

### Stretch (only if ahead)
- Congestion-prediction module (GAT+LSTM) as an alternative way to handle stale state
- Multi-agent / distributed variant for scalability
- Physical demo (OVS on Raspberry Pis / OpenFlow switch)
- Shadow-mode run against real telemetry from a lab network (**only with written permission**)

---

## 9. Repo layout

```
stale-but-safe-te/
├── README.md                # quickstart, results table, demo GIF
├── pyproject.toml / lockfile
├── Makefile                 # sim | train | eval | demo | reproduce-figN | test
├── docker/                  # Dockerfiles + docker-compose.yml
├── configs/                 # YAML: topologies, traffic, telemetry profiles, hyperparams
├── data/{topologies,traffic}/
├── src/
│   ├── env/                 # Gym env + telemetry impairment wrapper
│   ├── models/              # GNN encoder, policy/value heads
│   ├── train/               # PPO, curriculum, domain randomization
│   ├── baselines/           # SPF, ECMP, greedy, tuned heuristic, LP oracle, published baseline
│   ├── guard/               # 3-layer safety guard
│   ├── controller/          # Ryu app
│   └── eval/                # metrics, stats (bootstrap, rliable-style), plotting
├── experiments/             # one folder per experiment: config + results + notes
├── tests/
├── dashboard/
├── docs/{eval_protocol.md, architecture.md, howto/}
└── paper/                   # LaTeX/Overleaf sources, figure scripts
```

Conventions: every result tied to a git commit hash + config file; fixed seeds in configs; PRs reviewed by a teammate; no results from uncommitted code.

---

## 10. Testing strategy

- **Unit:** observation builder, impairment wrapper (verify delay/noise/drop statistics), reward, path computation, flow-rule translation, guard logic.
- **Property tests:** split ratios sum to 1; no loops; every demand has a valid path; guard never emits an invalid action.
- **Simulator regression:** fixed seed → fixed baseline numbers.
- **Integration (CI/nightly):** short Mininet + controller run; assert connectivity and ≥ X% of ECMP performance.
- **Chaos:** link kills, stale/missing stats, controller restart, inference timeout → fallback engages, traffic keeps flowing.
- **Soak:** 30–60 min, monitor controller CPU/memory.

---

## 11. Paper structure

1. **Abstract**
2. **Introduction:** the perfect-telemetry assumption, the three contributions, the claim
3. **Related Work:** (a) GNN/DRL routing and TE, (b) SDN TE and load balancing, (c) robustness/noise in RL control (including traffic-signal-control noise-robustness), (d) safe RL / fallback policies in systems, (e) benchmarking RL vs heuristics
4. **Problem Formulation:** graph, demands, telemetry model, MDP
5. **Method:** GNN-PPO with history + mask, telemetry randomization, safety guard
6. **Experimental Setup:** topologies, traffic, impairment profiles, baselines, metrics, statistics
7. **Results:** RQ1 degradation → RQ2 recovery → RQ3 guard Pareto → RQ4 generalization and sim→Mininet
8. **Discussion & Limitations:** emulation ≠ production, empirical (not formal) safety, scaled-down capacities
9. **Conclusion & Future Work**
10. **Artifact / Reproducibility statement**

**Planned key figures:** (1) degradation vs telemetry delay for all methods; (2) gain-vs-p99-regret Pareto from ε sweep; (3) zero-shot results across held-out topologies (performance profile); (4) sim vs Mininet agreement; (5) ablation bars.

**Venue targets (realistic for a student team):** a networking workshop or conference such as IEEE NetSoft, CNSM, NOMS, or ICC/Globecom workshops, or a journal such as IEEE Access / Computer Networks. Check each venue's current deadlines, page limits and fit before choosing. Talk to your faculty advisor about target and authorship early.

---

## 12. Reframing the 24-paper literature set

- **Keep (core):** the GNN+DRL routing set (Papers 2, 7, 14, 16, 21), SDN TE (5, 8, 22, 24), surveys (1, 12, 13).
- **Shrink to a sentence or drop:** Papers 3, 4, 10, 23 (membrane fouling, land cover, lncRNA, ITS slicing). Reviewers will see these as padding. Only keep one if you actually use its technique (e.g. 4's GAT+LSTM if you build the stretch predictor).
- **Keep as analogy only:** traffic-signal-control papers (15, 17–20), valuable specifically for the **noise-robustness** and reward-design angle.
- **Add:** ENERO, CFR-RL, PPO-R, GTSR, GRL-RR, Teal (WAN TE), *Towards Safe Online RL in Computer Systems*, RLScale-Bench, the noisy-state-perturbation TSC paper, and the Subito paper on noisy delay measurements. Read each before citing; I only saw abstracts/snippets.
- Housekeeping: your prep doc says 24 papers, not 25; confirm the master list.

---

## 13. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Agent can't beat the tuned heuristic | Medium–High | Gate 2; oracle warm-start; pivot to benchmark framing |
| Someone already published the telemetry-robustness angle | Medium | Literature pass in Phase 0; shift emphasis to guard + benchmark |
| Sim→Mininet mismatch | High | Validate early; domain randomization; claim only what Mininet confirms |
| Ryu/Mininet/PyTorch version pain | High | Docker + pinned versions in Phase 0 |
| Route flapping / reordering | Medium | Hysteresis, churn penalty, per-flow pinning |
| Overclaiming safety | Medium | Say "empirical worst-case bound"; show the Pareto curve |
| Scope creep | High | Stretch list is gated; the claim in Section 1 is the filter |
| Training cost | Medium | Small topologies first; vectorized envs; Colab/Kaggle GPUs |
| Weak baselines called out by reviewers | Medium | Equal tuning budget; LP oracle; published baseline re-implementation |

---

## 14. Do this week

1. **Literature pass (1–2 days).** Search Google Scholar / arXiv / IEEE Xplore for: "stale telemetry" + reinforcement learning + traffic engineering; "observation delay" + SDN + DRL routing; "partial observability" + GNN routing; "safe reinforcement learning" + traffic engineering + SDN; "fallback" + learned routing. Read the closest 5–10 and write down exactly how yours differs.
2. Create the repo, Docker image and CI; get a static Ryu app running in Mininet.
3. Download Topology Zoo graphs; fix the train/validation/test split now and don't touch the test set.
4. Write `docs/eval_protocol.md` (methods, impairment profiles, seeds, metrics, stats tests) before any experiments.
5. Build the env skeleton and impairment wrapper with SPF/ECMP baselines.
6. Get a faculty advisor's opinion on the claim and target venue.

---

## 15. Ethics and safety notes

- Never run experiments on a campus or production network without written permission; use shadow mode for any real telemetry.
- Don't publish raw traffic traces that could contain sensitive data; use public datasets or synthetic data.
- State clearly in the paper that results are from simulation/emulation and don't certify production safety.
