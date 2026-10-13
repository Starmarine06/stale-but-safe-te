| model | telemetry | dScore (mean over 24 graphs) | p99 regret | graphs better than tuned | delta vs full model |
|---|---|---|---|---|---|
| full model (seed 0) | ideal | -0.064 | 0.017 | 9 of 24 | - |
| no history (H=1) | ideal | -0.062 | 0.022 | 6 of 24 | +0.003 [+0.001, +0.004] |
| no telemetry-age input | ideal | -0.064 | 0.018 | 8 of 24 | +0.001 [+0.000, +0.001] |
| no GNN (per-link MLP) | ideal | -0.069 | 0.019 | 10 of 24 | -0.005 [-0.008, -0.002] |
| no fluid-model features | ideal | -0.056 | 0.019 | 8 of 24 | +0.008 [+0.005, +0.012] |
| full model (seed 0) | degraded | -0.060 | 0.019 | 21 of 24 | - |
| no history (H=1) | degraded | -0.060 | 0.019 | 21 of 24 | +0.000 [-0.000, +0.001] |
| no telemetry-age input | degraded | -0.061 | 0.020 | 22 of 24 | -0.001 [-0.001, -0.000] |
| no GNN (per-link MLP) | degraded | -0.065 | 0.024 | 23 of 24 | -0.004 [-0.007, -0.002] |
| no fluid-model features | degraded | -0.047 | 0.030 | 19 of 24 | +0.013 [+0.008, +0.019] |
