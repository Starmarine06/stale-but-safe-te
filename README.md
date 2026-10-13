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

## Headline finding (real traces, load 0.8, learned model trained only on synthetic graphs)
| network | oracle gain | learned | tuned heuristic | tuned under degraded telemetry |
|---|---|---|---|---|
| Abilene | 57% | 31% | 33% | p99 regret +0.35 (worse than ECMP) |
| GEANT | 58% | 42% | 38% | no worse than ECMP |
| Germany17 | 37% | 15% | 19% | -0.2%, p99 regret +0.14 |

Gain = reduction of mean max-link-utilisation vs ECMP. The ranking between tuned and learned depends on the
telemetry quality; see `experiments/final/real_sanity_load*.txt` and the paper draft for the full protocol.

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
