| network (test split) @load | telemetry | tuned | DOTE-style (ideal-trained) | DOTE-style (degraded-trained) | ours GNN, mean+-SD of 4 seeds | static LP (mean TM) | winner |
|---|---|---|---|---|---|---|---|
| abilene-test@0.6 | ideal | 28.5 / -0.018 | 48.5 / -0.094 | 49.0 / -0.080 | 24.9+-2.6 | 34.5 | dote-degraded |
| abilene-test@0.6 | snmp | 19.1 / +0.126 | 48.2 / -0.080 | 48.9 / -0.079 | 25.2+-3.4 | 34.5 | dote-degraded |
| abilene-test@0.6 | degraded | 28.4 / +0.049 | 48.0 / -0.079 | 48.8 / -0.079 | 25.2+-4.1 | 34.5 | dote-degraded |
| abilene-test@0.8 | ideal | 33.6 / -0.057 | 48.6 / -0.128 | 49.1 / -0.108 | 26.5+-2.2 | 34.5 | dote-degraded |
| abilene-test@0.8 | snmp | 27.7 / -0.025 | 48.2 / -0.112 | 49.1 / -0.109 | 26.1+-3.2 | 34.5 | dote-degraded |
| abilene-test@0.8 | degraded | 23.5 / +0.176 | 47.9 / -0.112 | 48.9 / -0.107 | 25.8+-3.4 | 34.5 | dote-degraded |
| abilene-test@1.0 | ideal | 36.9 / -0.084 | 48.5 / -0.159 | 49.0 / -0.137 | 27.6+-1.8 | 34.5 | dote-degraded |
| abilene-test@1.0 | snmp | 31.2 / +0.052 | 48.2 / -0.145 | 49.0 / -0.137 | 26.7+-3.0 | 34.5 | dote-degraded |
| abilene-test@1.0 | degraded | 21.4 / +0.323 | 47.8 / -0.144 | 48.8 / -0.134 | 26.4+-3.0 | 34.5 | dote-degraded |
| geant-test@0.6 | ideal | 16.5 / +0.046 | 18.0 / -0.025 | 17.8 / -0.026 | 15.5+-1.5 | -38.6 | dote-ideal |
| geant-test@0.6 | snmp | 9.2 / +0.108 | 18.0 / -0.023 | 17.8 / -0.023 | 15.3+-0.9 | -38.6 | dote-ideal |
| geant-test@0.6 | degraded | 2.2 / +0.296 | 17.8 / -0.023 | 17.7 / -0.022 | 14.2+-0.7 | -38.6 | dote-ideal |
| geant-test@0.8 | ideal | 20.8 / -0.016 | 18.2 / -0.026 | 17.9 / -0.031 | 16.7+-1.0 | -38.6 | tuned |
| geant-test@0.8 | snmp | 13.2 / +0.190 | 18.2 / -0.025 | 17.9 / -0.029 | 16.2+-0.6 | -38.6 | dote-ideal |

(cells: gain % vs ECMP / p99 regret)
