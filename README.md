# Stale-but-Safe TE

Delay-robust and safety-guarded learned traffic engineering for SDN. A GNN policy picks split ratios over
`k` candidate paths per demand from **impaired telemetry** (polling period, delay, noise, missing counters);
a three-layer runtime **guard** falls back to ECMP. Everything is evaluated against tuned heuristics and an LP
oracle in a fluid simulator, and re-validated in Mininet + Open vSwitch with a Ryu controller.

> Status: research code. Numbers below are produced by the scripts in `experiments/`; see
> `docs/eval_protocol.md` for the frozen protocol and `docs/related_work.md` for positioning
> (notably LOGGIA, arXiv 2604.02927, which already shows neural routers degrade under delay).

## Quickstart
```bash
pip install -e ".[dev]"          # core (numpy/scipy/networkx); add torch for the learned policy
python -m pytest -q              # 33 tests
stale-but-safe demo --topology abilene --mode shadow --model runs/ours_dfm_final.pt --profile snmp
stale-but-safe demo --topology Geant2012 --mode active --model runs/ours_dfm_final.pt --profile degraded
```

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
