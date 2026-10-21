# Clean-environment reproduction check (2026-10-14)

Procedure: `git clone` of the local repository into an empty directory, new virtual environment (Python 3.13, Windows 11), `pip install torch`
(CPU wheel) and `pip install -e ".[dev]"`, then:

| Check | Result |
|---|---|
| `python -m pytest -q` on the fresh clone (no real traces) | 38 passed, 1 skipped (the skip needs a prepared real trace) |
| Download `directed-nobel-germany-DFN-aggregated-5min-over-1day-native.tgz` from `https://sndlib.put.poznan.pl/download/` into `data/real/`; byte-identical to the archive used for the paper | yes |
| `python -m stale_but_safe.env.real prepare germany17`; then `pytest -q` | 38 passed, 1 skipped |
| `stale-but-safe audit --network germany17 --polling 5 --delay 3 --noise 0.2 --loss 0.1 --seeds 6` | runs; verdict "Use learned", headroom 36.5%, learned 11.8% [10.7, 13.0] |
| `experiments/final_eval.py` on 3 Zoo graphs, degraded telemetry, 5 seeds, compared with the stored `results_ours.json` | ECMP, heuristic and unguarded learned dScore identical to 4 decimals; the guarded variants differ by up to 0.0012 because the check used the default guard settings, not the headline guard |

Not checked: the other four SNDlib archives (download URLs follow the same pattern), training from scratch (hours per model on CPU), and the Docker/Mininet/Ryu
emulation on a second machine. The `zib.de` mirror in older documentation returns an error page; use the Poznan URL. A public repository URL does not exist yet.

# Second check (2026-10-22): fresh clone, tests and table regeneration

Procedure: `git clone` of the local repository at commit 9533d06 into an empty directory, new virtual environment (Python 3.13, Windows 11), CPU torch wheel, `pip install -e ".[dev]"`.

| Check | Result |
|---|---|
| `python -m pytest -q` | 42 passed, 1 skipped (the skip needs a prepared real trace) |
| `python experiments/reproduce_tables.py` (regenerates nine tables from the committed raw result files and compares them with the committed copies) | all nine identical |

This check found and fixed one stale artefact: the committed `table_dote.md` lacked four rows that the raw files produce, and was regenerated.

Still not re-run from scratch: simulations behind the raw result files (hours on CPU), the delay grid, the guard audit, the emulation runs (Docker, needs an otherwise idle machine) and the real-trace preparation. The clone was made from the local repository because the commit had not been pushed; repeat the two commands above against the GitHub URL after pushing.

