dScore vs ECMP (lower = better); 24 held-out Zoo graphs; learned = mean of 4 training seeds. 'diff' = heuristic - learned, so positive = learned better;
95% bootstrap CI over graphs. W/T/L = graphs where the learned policy wins/ties(|diff|<0.002)/loses. Re-tuned = 5 independent tuner seeds x 80 trials.

| profile | heuristic | dScore | p99 regret | diff vs learned [95% CI] | learned W/T/L |
|---|---|---|---|---|---|
| ideal | learned (4 seeds) | -0.065 | 0.018 | - | - |
| ideal | original (40 trials, 1 run) | -0.080 | 0.024 | -0.016 [-0.030, -0.005] | 1/10/13 |
| ideal | re-tuned, mean over 5 tuner seeds | -0.076 | 0.036 | -0.011 [-0.026, +0.002] | 4/11/9 |
| ideal | re-tuned, selected by tuning score | -0.085 | 0.026 | -0.020 [-0.036, -0.008] | 1/10/13 |
| ideal | re-tuned, per tuner seed (dScore) | -0.078, -0.085, -0.072, -0.078, -0.065 | | | |
| rich | learned (4 seeds) | -0.063 | 0.018 | - | - |
| rich | original (40 trials, 1 run) | -0.067 | 0.033 | -0.004 [-0.014, +0.004] | 6/12/6 |
| rich | re-tuned, mean over 5 tuner seeds | -0.076 | 0.036 | -0.014 [-0.027, -0.004] | 3/10/11 |
| rich | re-tuned, selected by tuning score | -0.073 | 0.041 | -0.010 [-0.024, +0.002] | 4/10/10 |
| rich | re-tuned, per tuner seed (dScore) | -0.080, -0.081, -0.073, -0.073, -0.076 | | | |
| snmp | learned (4 seeds) | -0.062 | 0.017 | - | - |
| snmp | original (40 trials, 1 run) | -0.049 | 0.048 | +0.013 [+0.003, +0.023] | 9/12/3 |
| snmp | re-tuned, mean over 5 tuner seeds | -0.060 | 0.077 | +0.002 [-0.007, +0.012] | 7/10/7 |
| snmp | re-tuned, selected by tuning score | -0.072 | 0.056 | -0.010 [-0.023, +0.000] | 3/11/10 |
| snmp | re-tuned, per tuner seed (dScore) | -0.056, -0.072, -0.060, -0.060, -0.053 | | | |
| degraded | learned (4 seeds) | -0.061 | 0.019 | - | - |
| degraded | original (40 trials, 1 run) | +0.014 | 0.183 | +0.075 [+0.036, +0.120] | 14/10/0 |
| degraded | re-tuned, mean over 5 tuner seeds | -0.029 | 0.092 | +0.033 [+0.018, +0.049] | 15/9/0 |
| degraded | re-tuned, selected by tuning score | -0.040 | 0.093 | +0.021 [+0.010, +0.034] | 14/9/1 |
| degraded | re-tuned, per tuner seed (dScore) | -0.021, -0.040, -0.045, -0.034, -0.004 | | | |
