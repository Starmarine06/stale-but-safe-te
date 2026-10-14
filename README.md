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

* **The ranking depends on telemetry quality.** 24 held-out Topology Zoo graphs, 4 training seeds, heuristic re-tuned (5 tuner seeds x 80
  trials per profile): the heuristic is better or equal on average with ideal, rich and SNMP telemetry (SNMP: tie, +0.002 [-0.007, +0.012]); the
  learned policy is ahead only under heavily degraded telemetry (+0.033 [+0.018, +0.049], 15 wins / 9 ties / 0 losses) and has lower p99
  regret in every profile (0.017-0.019 vs 0.036-0.092). Earlier claims of learned wins under SNMP came from a weakly tuned baseline
  (`experiments/final/table_retune.md`). Real traces with the re-tuned heuristic (`table_realci_v2.md`, 36 cells, window-level paired CIs): learned ahead
  in 12, heuristic in 17, tie 7; by profile ideal 0/9/0, rich 0/6/3, SNMP 3/2/4, degraded 9/0/0 (learned/heuristic/tie). Stress scenarios: learned ahead
  under degraded telemetry in all four (`table_v2_guard_stress.md`). The guard roughly halves p99 regret of both policies at +0.011 mean dScore cost for the learned one.
* **Staleness (delay, polling period) matters; noise and missing counters hardly do.** A heuristic with frozen parameters
  collapses with delay, but one re-tuned per condition stays close. In a delay x polling grid with five independent tuning runs
  per cell, the re-tuned heuristic is competitive with fresh, frequent telemetry and falls behind as delay and polling period
  grow. Paired window-level intervals (16 fresh windows): on GEANT learned ahead in 6 of 9 cells (2.5-12.3 points), tie in 2,
  heuristic ahead in 1; on Abilene a 16-window sample was unresolved, and 48 further windows give the same gradient as GEANT (heuristic ahead at d<=3 with P=1, learned ahead in 3 cells with stale or infrequent telemetry, 4 ties; `table_abilene_more.md`). An audit (`experiments/final/table_abilene_audit.md`) shows the Abilene sign moves with the window sample and with the
  heuristic's tuning, with no implementation inconsistency found. Not a sharp crossover.
* **It is conditional.** A DOTE-style model trained on a network's own history beats the zero-shot policy (Abilene 48-49% vs
  22-25%); fine-tuning on history recovers most of it. Fluid-model features carry the result, not graph message passing. In
  overloaded trace windows the reactive heuristic wins (45-49% vs 9-21%).
* **Guard:** strong for link failures (catches ~90% of harmful proposals), weak for degraded telemetry (~22%); no formal guarantee;
  costs more in emulation than in simulation.
* **Operational cost:** the learned policy reprograms ~2x more flows than the re-tuned heuristic (270-332 vs ~139 per minute; ~10x more than the original, weaker heuristic); a 5-20% update threshold removes 90-99%
  of updates in simulation at no loss of gain (77% in emulation for 0.018 max-util). Decision time 2 ms (12 nodes) to 311 ms (80 nodes).
* **Emulation (28 runs, 4 scenarios):** simulated vs emulated mean max-util correlation 0.99, mean abs difference 0.017.
* **Audit tool:** `stale-but-safe audit` gives headroom, per-method gains with 95% intervals, and one of: stay on ECMP / use X /
  insufficient evidence. An offline estimate, not a deployment recommendation.

## Data and availability
Topology Zoo graphs are not redistributed here; fetch them with `python -m stale_but_safe.env.zoo fetch` (downloads the public GitHub mirror, 263 GraphML files) before running anything that uses Zoo graphs. Please cite Knight et al., *The Internet Topology Zoo*, IEEE JSAC 2011.

Repository: https://github.com/Starmarine06/stale-but-safe-te (currently private; make it public before submission). The SNDlib archives are public and not redistributed; download them from
`https://sndlib.put.poznan.pl/download/<archive>.tgz` (archive names in `src/stale_but_safe/env/real.py` and `docs/howto.md`) into `data/real/`, then run
`python -m stale_but_safe.env.real prepare <name>`. A clean-environment check is in `docs/reproduction_check.md`. The default heuristic parameters for the audit tool
are `configs/tuned_heuristic_v2.yaml` (re-tuned, 5 tuner seeds x 80 trials); `configs/tuned_heuristic.yaml` is the original 40-trial heuristic kept for
reproducing earlier tables.

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

## Paper artefact map
Each paper table is produced by one script and one result file under `experiments/final/`.

| Paper item | Script(s) | Result file |
|---|---|---|
| Table 1, Table A1, Fig. 1 | `experiments/retune_profiles.py`, `experiments/analyze_retune.py` | `table_retune.md` |
| Table 2, Table 10 | `experiments/final_eval.py --tuned-only`, `experiments/analyze_v2_guard_stress.py` | `table_v2_guard_stress.md` |
| Table 3, Fig. 2 | `experiments/realci_v2.py`, `experiments/analyze_realci_v2.py` | `table_realci_v2.md` |
| Table 4 | `experiments/delay_controlled.py 1` | `table_delay_part1.md` |
| Table 5 | `experiments/delay_tuning_repeat.py` | `table_delay_repeat.md` |
| Table 6, Fig. 3 | `experiments/delay_grid_ci.py` | `table_delay_ci.md` |
| Table 7 | `experiments/abilene_audit.py` | `table_abilene_audit.md` |
| Table 8 | `experiments/overload_windows.py abilene 0.8 configs/tuned_heuristic_v2.yaml _v2` | `table_overload_v2.md` |
| Table 9 | `experiments/guard_audit.py` | `table_guard_audit.md` |
| Table 11 | `docs/howto.md` (emulation), `experiments/emu_analyze2.py`, `experiments/emu_analyze3.py` | `table_emu2.md`, `table_emu3.md` |
| Table 12 | `experiments/finetune_real.py`, `experiments/dote_baseline.py` | `table_ft.md`, `table_dote.md` |
| Table 13 | `experiments/scaling_sensitivity.py configs/tuned_heuristic_v2.yaml` | `table_scaling_v2.md` |
| GNN versus MLP, ablations | `experiments/analyze_gnn_vs_mlp.py`, `experiments/analyze_ablations.py` | `table_gnn_vs_mlp.md`, `table_ablations.md` |
| Loss-weight sensitivity | `experiments/sensitivity_lossweight.py experiments/final/results_v2_zoo.json` | `table_lossweight_v2.md` |

Build the paper with `python paper/make_figures.py` and `python paper/md2tex.py` followed by `bash paper/tnsm/build.sh` (needs a LaTeX installation with IEEEtran).

## Layout
See `docs/architecture.md`. Short version: `src/stale_but_safe/{env,baselines,models,train,guard,eval,emu,controller}`,
`experiments/` (one script per result), `docker/` (Mininet and Ryu images), `docs/`, `paper/`.

## Limitations
Fluid model (no TCP/packet dynamics); empirical (not formal) safety bound; Mininet capacities scaled down;
real SNDlib traces are scaled to a target load and telemetry faults are modelled, not measured; Topology Zoo graphs have no real capacities (degree-based rule); results from simulation
and emulation do not certify production safety. Never run experiments on a campus or production network
without written permission; use shadow mode for real telemetry.
