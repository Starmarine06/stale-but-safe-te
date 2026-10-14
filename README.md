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
Full write-up: `paper/paper.md`; protocol deviations: `docs/eval_protocol.md`. Real SNDlib traces (Abilene, GEANT,
Germany17, Germany50, Brain) are scaled to a target load; telemetry faults are modelled, not measured.

* **The ranking depends on telemetry quality.** 24 held-out Topology Zoo graphs, 4 training seeds: tuned heuristic better with
  clean telemetry, learned policy better with degraded telemetry (21-23 of 24 graphs). Real traces, 10 fresh windows per cell:
  learned never behind the heuristic under rich / SNMP / degraded telemetry; clean telemetry is mixed (4 / 3 / 2 of 9 cells).
* **Staleness (delay, polling period) matters; noise and missing counters hardly do.** A heuristic with frozen parameters
  collapses with delay, but one re-tuned per condition stays close. In a delay x polling grid with five independent tuning runs
  per cell, the re-tuned heuristic is competitive with fresh, frequent telemetry and falls behind as delay and polling period
  grow. Paired window-level intervals (16 fresh windows): on GEANT learned ahead in 6 of 9 cells (2.5-12.3 points), tie in 2,
  heuristic ahead in 1; on Abilene unresolved (intervals include 0 in 6 of 9 cells, heuristic ahead in 3). Not a sharp crossover.
* **It is conditional.** A DOTE-style model trained on a network's own history beats the zero-shot policy (Abilene 48-49% vs
  22-25%); fine-tuning on history recovers most of it. Fluid-model features carry the result, not graph message passing. In
  overloaded trace windows the reactive heuristic wins (45-49% vs 9-21%).
* **Guard:** strong for link failures (catches ~90% of harmful proposals), weak for degraded telemetry (~22%); no formal guarantee;
  costs more in emulation than in simulation.
* **Operational cost:** the learned policy reprograms ~10x more flows than the heuristic; a 5-20% update threshold removes 90-99%
  of updates in simulation at no loss of gain (77% in emulation for 0.018 max-util). Decision time 2 ms (12 nodes) to 311 ms (80 nodes).
* **Emulation (28 runs, 4 scenarios):** simulated vs emulated mean max-util correlation 0.99, mean abs difference 0.017.
* **Audit tool:** `stale-but-safe audit` gives headroom, per-method gains with 95% intervals, and one of: stay on ECMP / use X /
  insufficient evidence. An offline estimate, not a deployment recommendation.

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
