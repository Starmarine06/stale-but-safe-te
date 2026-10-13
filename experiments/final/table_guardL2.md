
### profile = ideal  (n_topologies = 24)

| method | dScore (95% CI) | IQM | gain % | p99 regret | loss | fallback |
|---|---|---|---|---|---|---|
| ecmp | +0.000 [+0.000, +0.000] | +0.000 | 0.0 | 0.000 | 0.013 | - |
| gnn-ours | -0.064 [-0.091, -0.039] | -0.051 | 4.3 | 0.017 | 0.008 | - |
| gnn-ours+guard | -0.054 [-0.079, -0.031] | -0.038 | 3.7 | 0.008 | 0.009 | 0.82 |
| oracle (n=12) | -0.086 [-0.153, -0.028] | -0.054 | 10.2 | -0.005 | 0.018 | - |
| tuned | -0.080 [-0.114, -0.049] | -0.061 | 5.6 | 0.024 | 0.008 | - |
| tuned+guard | -0.060 [-0.087, -0.035] | -0.043 | 4.4 | 0.011 | 0.010 | 0.80 |

Paired Wilcoxon on dScore (Holm-adjusted p): gnn-ours vs tuned: p=0.00664 (mean diff +0.016); gnn-ours+guard vs gnn-ours: p=7.26e-05 (mean diff +0.011); tuned+guard vs tuned: p=0.000119 (mean diff +0.020); gnn-ours vs ecmp: p=7.26e-05 (mean diff -0.064)

### profile = rich  (n_topologies = 24)

| method | dScore (95% CI) | IQM | gain % | p99 regret | loss | fallback |
|---|---|---|---|---|---|---|
| ecmp | +0.000 [+0.000, +0.000] | +0.000 | 0.0 | 0.000 | 0.013 | - |
| gnn-ours | -0.062 [-0.088, -0.038] | -0.049 | 4.1 | 0.018 | 0.009 | - |
| gnn-ours+guard | -0.051 [-0.076, -0.029] | -0.035 | 3.6 | 0.009 | 0.010 | 0.82 |
| oracle (n=12) | -0.086 [-0.153, -0.028] | -0.054 | 10.2 | -0.005 | 0.018 | - |
| tuned | -0.067 [-0.095, -0.040] | -0.053 | 4.5 | 0.033 | 0.008 | - |
| tuned+guard | -0.054 [-0.078, -0.031] | -0.039 | 3.8 | 0.017 | 0.010 | 0.79 |

Paired Wilcoxon on dScore (Holm-adjusted p): gnn-ours vs tuned: p=0.841 (mean diff +0.005); gnn-ours+guard vs gnn-ours: p=0.000178 (mean diff +0.011); tuned+guard vs tuned: p=0.000178 (mean diff +0.013); gnn-ours vs ecmp: p=0.000108 (mean diff -0.062)

### profile = snmp  (n_topologies = 24)

| method | dScore (95% CI) | IQM | gain % | p99 regret | loss | fallback |
|---|---|---|---|---|---|---|
| ecmp | +0.000 [+0.000, +0.000] | +0.000 | 0.0 | 0.000 | 0.013 | - |
| gnn-ours | -0.062 [-0.088, -0.037] | -0.049 | 4.1 | 0.017 | 0.009 | - |
| gnn-ours+guard | -0.051 [-0.075, -0.029] | -0.036 | 3.5 | 0.010 | 0.010 | 0.82 |
| oracle (n=12) | -0.086 [-0.153, -0.028] | -0.054 | 10.2 | -0.005 | 0.018 | - |
| tuned | -0.049 [-0.071, -0.029] | -0.038 | 3.2 | 0.048 | 0.009 | - |
| tuned+guard | -0.039 [-0.057, -0.023] | -0.030 | 2.7 | 0.023 | 0.010 | 0.87 |

Paired Wilcoxon on dScore (Holm-adjusted p): gnn-ours vs tuned: p=0.0261 (mean diff -0.012); gnn-ours+guard vs gnn-ours: p=0.000262 (mean diff +0.011); tuned+guard vs tuned: p=0.00107 (mean diff +0.010); gnn-ours vs ecmp: p=0.000108 (mean diff -0.062)

### profile = degraded  (n_topologies = 24)

| method | dScore (95% CI) | IQM | gain % | p99 regret | loss | fallback |
|---|---|---|---|---|---|---|
| ecmp | +0.000 [+0.000, +0.000] | +0.000 | 0.0 | 0.000 | 0.013 | - |
| gnn-ours | -0.060 [-0.086, -0.037] | -0.048 | 4.0 | 0.019 | 0.009 | - |
| gnn-ours+guard | -0.049 [-0.072, -0.028] | -0.034 | 3.4 | 0.008 | 0.010 | 0.83 |
| oracle (n=12) | -0.086 [-0.153, -0.028] | -0.054 | 10.2 | -0.005 | 0.018 | - |
| tuned | +0.014 [-0.023, +0.057] | -0.005 | 0.6 | 0.183 | 0.017 | - |
| tuned+guard | -0.034 [-0.052, -0.018] | -0.020 | 2.5 | 0.030 | 0.011 | 0.83 |

Paired Wilcoxon on dScore (Holm-adjusted p): gnn-ours vs tuned: p=0.000791 (mean diff -0.074); gnn-ours+guard vs gnn-ours: p=0.000343 (mean diff +0.011); tuned+guard vs tuned: p=0.00514 (mean diff -0.047); gnn-ours vs ecmp: p=0.000155 (mean diff -0.060)
