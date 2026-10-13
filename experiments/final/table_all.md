
### profile = ideal  (n_topologies = 24)

| method | dScore (95% CI) | IQM | gain % | p99 regret | loss | fallback |
|---|---|---|---|---|---|---|
| ecmp | +0.000 [+0.000, +0.000] | +0.000 | 0.0 | 0.000 | 0.013 | - |
| gnn-ideal | -0.066 [-0.094, -0.040] | -0.052 | 4.4 | 0.016 | 0.008 | - |
| gnn-ideal+guard | -0.064 [-0.091, -0.039] | -0.050 | 4.4 | 0.013 | 0.009 | 0.32 |
| gnn-ours | -0.064 [-0.091, -0.039] | -0.051 | 4.3 | 0.017 | 0.008 | - |
| gnn-ours+guard | -0.062 [-0.089, -0.038] | -0.049 | 4.2 | 0.013 | 0.009 | 0.34 |
| oracle (n=12) | -0.086 [-0.153, -0.028] | -0.054 | 10.2 | -0.005 | 0.018 | - |
| tuned | -0.080 [-0.114, -0.049] | -0.061 | 5.6 | 0.024 | 0.008 | - |
| tuned+guard | -0.076 [-0.109, -0.046] | -0.056 | 5.3 | 0.023 | 0.008 | 0.06 |

Paired Wilcoxon on dScore (Holm-adjusted p): gnn-ours vs tuned: p=0.0133 (mean diff +0.016); gnn-ours vs gnn-ideal: p=0.00579 (mean diff +0.002); gnn-ours+guard vs gnn-ours: p=0.00217 (mean diff +0.002); tuned+guard vs tuned: p=0.0146 (mean diff +0.004); gnn-ours vs ecmp: p=9.07e-05 (mean diff -0.064)

### profile = rich  (n_topologies = 24)

| method | dScore (95% CI) | IQM | gain % | p99 regret | loss | fallback |
|---|---|---|---|---|---|---|
| ecmp | +0.000 [+0.000, +0.000] | +0.000 | 0.0 | 0.000 | 0.013 | - |
| gnn-ideal | -0.064 [-0.091, -0.039] | -0.051 | 4.3 | 0.017 | 0.009 | - |
| gnn-ideal+guard | -0.062 [-0.089, -0.038] | -0.049 | 4.2 | 0.015 | 0.009 | 0.33 |
| gnn-ours | -0.062 [-0.088, -0.038] | -0.049 | 4.1 | 0.018 | 0.009 | - |
| gnn-ours+guard | -0.060 [-0.086, -0.036] | -0.047 | 4.1 | 0.016 | 0.009 | 0.33 |
| oracle (n=12) | -0.086 [-0.153, -0.028] | -0.054 | 10.2 | -0.005 | 0.018 | - |
| tuned | -0.067 [-0.095, -0.040] | -0.053 | 4.5 | 0.033 | 0.008 | - |
| tuned+guard | -0.064 [-0.091, -0.038] | -0.049 | 4.3 | 0.029 | 0.009 | 0.07 |

Paired Wilcoxon on dScore (Holm-adjusted p): gnn-ours vs tuned: p=0.841 (mean diff +0.005); gnn-ours vs gnn-ideal: p=0.00468 (mean diff +0.002); gnn-ours+guard vs gnn-ours: p=0.00408 (mean diff +0.002); tuned+guard vs tuned: p=0.0384 (mean diff +0.003); gnn-ours vs ecmp: p=0.000135 (mean diff -0.062)

### profile = snmp  (n_topologies = 24)

| method | dScore (95% CI) | IQM | gain % | p99 regret | loss | fallback |
|---|---|---|---|---|---|---|
| ecmp | +0.000 [+0.000, +0.000] | +0.000 | 0.0 | 0.000 | 0.013 | - |
| gnn-ideal | -0.062 [-0.088, -0.038] | -0.049 | 4.1 | 0.021 | 0.009 | - |
| gnn-ideal+guard | -0.058 [-0.083, -0.035] | -0.045 | 4.0 | 0.018 | 0.009 | 0.36 |
| gnn-ours | -0.062 [-0.088, -0.037] | -0.049 | 4.1 | 0.017 | 0.009 | - |
| gnn-ours+guard | -0.058 [-0.084, -0.035] | -0.045 | 4.0 | 0.015 | 0.009 | 0.36 |
| oracle (n=12) | -0.086 [-0.153, -0.028] | -0.054 | 10.2 | -0.005 | 0.018 | - |
| tuned | -0.049 [-0.071, -0.029] | -0.038 | 3.2 | 0.048 | 0.009 | - |
| tuned+guard | -0.046 [-0.065, -0.028] | -0.037 | 3.1 | 0.028 | 0.010 | 0.10 |

Paired Wilcoxon on dScore (Holm-adjusted p): gnn-ours vs tuned: p=0.0784 (mean diff -0.012); gnn-ours vs gnn-ideal: p=0.494 (mean diff +0.000); gnn-ours+guard vs gnn-ours: p=0.000805 (mean diff +0.003); tuned+guard vs tuned: p=0.267 (mean diff +0.003); gnn-ours vs ecmp: p=0.000135 (mean diff -0.062)

### profile = degraded  (n_topologies = 24)

| method | dScore (95% CI) | IQM | gain % | p99 regret | loss | fallback |
|---|---|---|---|---|---|---|
| ecmp | +0.000 [+0.000, +0.000] | +0.000 | 0.0 | 0.000 | 0.013 | - |
| gnn-ideal | -0.059 [-0.084, -0.036] | -0.048 | 3.9 | 0.023 | 0.009 | - |
| gnn-ideal+guard | -0.033 [-0.049, -0.019] | -0.023 | 2.2 | 0.004 | 0.011 | 0.70 |
| gnn-ours | -0.060 [-0.086, -0.037] | -0.048 | 4.0 | 0.019 | 0.009 | - |
| gnn-ours+guard | -0.034 [-0.050, -0.019] | -0.025 | 2.3 | 0.003 | 0.010 | 0.72 |
| oracle (n=12) | -0.086 [-0.153, -0.028] | -0.054 | 10.2 | -0.005 | 0.018 | - |
| tuned | +0.014 [-0.023, +0.057] | -0.005 | 0.6 | 0.183 | 0.017 | - |
| tuned+guard | -0.021 [-0.033, -0.009] | -0.010 | 1.5 | 0.057 | 0.012 | 0.70 |

Paired Wilcoxon on dScore (Holm-adjusted p): gnn-ours vs tuned: p=0.00119 (mean diff -0.074); gnn-ours vs gnn-ideal: p=0.377 (mean diff -0.001); gnn-ours+guard vs gnn-ours: p=8.34e-06 (mean diff +0.027); tuned+guard vs tuned: p=0.377 (mean diff -0.034); gnn-ours vs ecmp: p=0.000155 (mean diff -0.060)
