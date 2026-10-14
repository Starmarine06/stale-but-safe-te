### Zoo, guard L2-only eps=-0.06, 24 graphs. Heuristic = re-tuned (best tuning score). dScore vs ECMP, lower better; 95% CI over graphs

| profile | method | dScore | p99 regret | guard fallback rate |
|---|---|---|---|---|
| ideal | gnn-ours | -0.064 [-0.091, -0.039] | 0.017 | - |
| ideal | gnn-ours+guard | -0.054 [-0.079, -0.030] | 0.008 | 0.82 |
| ideal | tuned | -0.085 [-0.120, -0.052] | 0.026 | - |
| ideal | tuned+guard | -0.064 [-0.094, -0.038] | 0.008 | 0.84 |
| rich | gnn-ours | -0.062 [-0.088, -0.037] | 0.018 | - |
| rich | gnn-ours+guard | -0.051 [-0.076, -0.028] | 0.009 | 0.82 |
| rich | tuned | -0.073 [-0.105, -0.043] | 0.041 | - |
| rich | tuned+guard | -0.058 [-0.084, -0.032] | 0.019 | 0.81 |
| snmp | gnn-ours | -0.062 [-0.087, -0.037] | 0.017 | - |
| snmp | gnn-ours+guard | -0.051 [-0.075, -0.029] | 0.010 | 0.82 |
| snmp | tuned | -0.072 [-0.101, -0.043] | 0.056 | - |
| snmp | tuned+guard | -0.054 [-0.078, -0.030] | 0.031 | 0.78 |
| degraded | gnn-ours | -0.060 [-0.085, -0.036] | 0.019 | - |
| degraded | gnn-ours+guard | -0.049 [-0.073, -0.028] | 0.008 | 0.83 |
| degraded | tuned | -0.040 [-0.062, -0.018] | 0.093 | - |
| degraded | tuned+guard | -0.042 [-0.060, -0.024] | 0.030 | 0.82 |

### Stress scenarios (12 graphs, original eval protocol), learned vs re-tuned heuristic; diff = heuristic - learned (positive = learned better)

| scenario | profile | learned dScore | re-tuned heuristic dScore | diff [95% CI over graphs] | learned W/T/L | p99 regret learned / heuristic | original-heuristic diff |
|---|---|---|---|---|---|---|---|
| fail | snmp | -0.066 | -0.070 | -0.004 [-0.018, +0.009] | 3/6/3 | 0.018 / 0.065 | +0.021 |
| fail | degraded | -0.064 | -0.044 | +0.020 [+0.007, +0.036] | 6/6/0 | 0.019 / 0.081 | +0.056 |
| shift | snmp | -0.059 | -0.047 | +0.011 [+0.001, +0.027] | 4/7/1 | 0.044 / 0.101 | +0.032 |
| shift | degraded | -0.057 | -0.021 | +0.035 [+0.015, +0.058] | 7/5/0 | 0.045 / 0.143 | +0.078 |
| flash | snmp | -0.073 | -0.067 | +0.005 [-0.009, +0.023] | 5/4/3 | 0.022 / 0.078 | +0.036 |
| flash | degraded | -0.069 | -0.036 | +0.032 [+0.012, +0.059] | 7/5/0 | 0.028 / 0.105 | +0.074 |
| hotspot | snmp | -0.126 | -0.090 | +0.036 [-0.012, +0.105] | 4/2/6 | 0.042 / 0.116 | +0.066 |
| hotspot | degraded | -0.123 | -0.022 | +0.101 [+0.029, +0.204] | 9/3/0 | 0.048 / 0.167 | +0.190 |
