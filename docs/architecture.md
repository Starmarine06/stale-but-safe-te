# Architecture

```
            ┌────────────────────────── simulator (src/stale_but_safe/env) ─────────────────────────┐
 TrafficGen ─► true demand D_t ─► fluid model (inc @ x) ─► true link util u_t, loss, delay, churn
                                         ▲ splits (D,k)            │ push every interval
                                         │                         ▼
                                   Policy.act(obs) ◄── Channel(P, d, sigma, p) per stream
                                   (obs = impaired util, age, demand estimate, link_up)
            └───────────────────────────────────────────────────────────────────────────────────────┘

 Policies (src/stale_but_safe/baselines, models, guard)
   SPF, ECMP, Greedy, TunedHeuristic (per-profile tuned), LPOracle (peeks at the true state)
   LearnedPolicy: GNNPolicy + Featurizer (history H, age mask, model-based features)
   Guard(policy): L1 hard rules, L2 pre-action fluid check (eps), L3 runtime monitor with backoff

 Emulation (src/stale_but_safe/emu, controller)
   scenario.json ─► Mininet runner (OVS, tc-shaped links, paced UDP senders, counting receivers)
                 ─► Ryu app: OpenFlow port/flow stats ─► same Channel impairments ─► Policy(+Guard)
                    ─► select-group re-programming with hysteresis (VLAN-tagged candidate paths)
```

## Key design decisions
* **Topology-agnostic action.** Every demand has `k` candidate paths; the policy outputs a softmax
  over them. The GNN runs on the *line graph of links* and path scores come from pooling link
  embeddings along each path, so one set of weights works for any topology/size.
* **Impairment is a first-class component** (`env/telemetry.py::Channel`): polling period P,
  delay d, multiplicative log-normal noise sigma (mean preserving), drop probability p (the stale value
  is kept and its age grows). The same class is used by the simulator and the Ryu controller.
* **Controller knowledge is not telemetry.** The featurizer uses the controller's own last installed
  splits and the link capacities to compute fluid-model predictions (ECMP-predicted and
  current-config-predicted utilisation, per-path "what-if" bottleneck) from the *estimated* demand.
* **Training.** (1) DAgger-style distillation of a privileged teacher (two-phase LP on the true state)
  from the impaired observation, with telemetry randomised per episode; (2) PPO fine-tuning on the
  environment reward expressed as improvement over the ECMP counterfactual.
* **Guard** is policy-agnostic and always falls back to ECMP.
* **Emulation loop.** Interval = `delta` seconds. At each tick the controller reads real OpenFlow
  counters (true utilisation, jitter included), pushes them through `Channel`, runs the policy,
  and re-programs only the select groups whose split changed by more than `SBS_HYST`.
  Weighted splits use OpenFlow select groups; every demand is carried by 16 UDP sub-flows so per-flow
  hashing realises fractional splits.

## Where things are
| Concern | Path |
|---|---|
| Environment, telemetry, traffic | `src/stale_but_safe/env/` |
| Baselines incl. LP oracle, tuner | `src/stale_but_safe/baselines/heuristics.py`, `eval/tune.py` |
| GNN policy, features | `src/stale_but_safe/models/` |
| Training | `src/stale_but_safe/train/{bc,ppo,scenarios}.py` |
| Guard | `src/stale_but_safe/guard/guard.py` |
| Evaluation, statistics | `src/stale_but_safe/eval/{runner,compare,stats}.py` |
| Emulation | `src/stale_but_safe/emu/`, `controller/ryu_app.py`, `docker/` |
| Frozen protocol | `docs/eval_protocol.md` |
