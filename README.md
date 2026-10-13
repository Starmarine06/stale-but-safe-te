# Stale-but-Safe TE

**Is traffic engineering worth it on your network, and does it survive your telemetry?**

`stale-but-safe audit` answers that for a given topology, traffic and telemetry quality (polling period,
delay, noise, missing counters). It measures the best-case headroom over ECMP (LP oracle), how a tuned
heuristic, a learned GNN policy and a guarded version of each behave under *your* telemetry, and recommends
what to run, in one self-contained HTML report. A shadow mode lets a policy propose routes while ECMP keeps
running. Everything is evaluated in a fluid simulator on real SNDlib topologies and traffic traces
(Abilene, GEANT, Germany17) plus 263 Topology Zoo graphs, and re-validated in Mininet + Open vSwitch with a
Ryu controller.

> Status: research code. Numbers come from the scripts in `experiments/`; see `docs/eval_protocol.md` for the
> frozen protocol and `docs/related_work.md` for positioning (notably LOGGIA, arXiv 2604.02927, which already
> shows neural routers degrade under delay).

## Quickstart
```bash
pip install -e ".[dev]"          # core (numpy/scipy/networkx); add torch for the learned policy
python -m pytest -q              # 36 tests
# real data (public SNDlib): download the three *-native.tgz and network .txt files into data/real/, then
python -m stale_but_safe.env.real prepare
stale-but-safe audit --network abilene --polling 5 --delay 3 --noise 0.2 --loss 0.1 --out reports/abilene
stale-but-safe audit --network geant --profile snmp --out reports/geant
stale-but-safe audit --network my_topology.graphml --traffic my_traffic.csv --out reports/mine   # CSV: t,src,dst,value
stale-but-safe demo --topology abilene --mode shadow --model runs/ours_dfm_final.pt --profile snmp
```

## Headline findings
Real SNDlib traces (Abilene, GEANT, Germany17, Germany50, Brain), 3 loads x 4 telemetry qualities; the learned
policy is trained only on small synthetic graphs (4 training seeds). Full table: `experiments/final/table_real.md`.

* **Ranking depends on telemetry quality.** In the four networks with TE headroom the learned policy is ahead of the
  tuned heuristic in 36 of 36 cells with rich / SNMP / degraded telemetry; with clean telemetry it is 7 ahead, 4 behind,
  1 tie (the tuned heuristic wins on GEANT and Germany17 at loads 0.8-1.0, by 1.5-4.1 points).
* **Tuned heuristics can fail badly under degraded telemetry.** Abilene at load 1.0: -4.0% vs ECMP (worst-case regret
  +1.02) while the learned policy gains 29%; Germany17 -3%. Not universal: Germany50 and GEANT do not collapse.
* **Held-out Topology Zoo (24 graphs, 4 seeds):** learned beats tuned on 21-23 of 24 graphs under degraded telemetry;
  seed-to-seed spread 0.001-0.002 in score.
* **Guard (validation-chosen, L2-only):** keeps ~82% of the learned gain on Zoo graphs and halves worst-case regret
  (0.019 -> 0.008); costs under 1 point on real traces; rescues the tuned heuristic.
* **Emulation (Mininet + OVS + Ryu):** time-series correlation with the simulator 0.90-0.98; costs: ~10x more
  flow-mods than the heuristic.
* **Honest limits:** the learned policy gets about half of the oracle headroom; traffic scaled to a target load;
  telemetry faults modelled, not measured; Brain's benefit is confined to rare windows. See `paper/paper.md`.

## Reproduce
| Step | Command |
|---|---|
| Tune baselines per telemetry profile | `python -m stale_but_safe.eval.tune` |
| Train ours (randomised telemetry) | `python -m stale_but_safe.train.dfm --mode random --iters 12 --out runs/ours.pt` then `--init runs/ours.pt --iters 4 --lr 2e-4` |
| Train ideal-telemetry baseline | same with `--mode ideal` |
| Ablations | `--H 1`, `--no-age`, `--no-gnn`, `--no-model` |
| Held-out evaluation | `python experiments/final_eval.py --ours runs/ours_dfm_final.pt --out experiments/final/results_ours.json` |
| Tables + statistics | `python experiments/analyze_final.py experiments/final/results_*.json` |
| Guard sweeps | `python experiments/guard_sweep.py <model>`, `python experiments/guard_tuned.py` |
| Emulation | see `docs/howto.md` (needs Docker; Linux containers; privileged Mininet) |

Training uses a differentiable fluid model (DOTE-style direct optimisation of the evaluation score) with the
true next-interval demand as privileged information; the policy only ever sees the impaired observation.
Earlier attempts (LP-teacher distillation, PPO fine-tuning) are kept in `train/bc.py`, `train/ppo.py` and
reported as negative results in the paper draft.

## Layout
See `docs/architecture.md`. Short version: `src/stale_but_safe/{env,baselines,models,train,guard,eval,emu,controller}`,
`experiments/` (one script per result), `docker/` (Mininet and Ryu images), `docs/`, `paper/`.

## Limitations
Fluid model (no TCP/packet dynamics); empirical (not formal) safety bound; Mininet capacities scaled down;
synthetic traffic only; Topology Zoo graphs have no real capacities (degree-based rule); results from simulation
and emulation do not certify production safety. Never run experiments on a campus or production network
without written permission; use shadow mode for real telemetry.
