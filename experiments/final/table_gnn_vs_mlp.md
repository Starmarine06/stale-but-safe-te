| telemetry | model | training seeds | dScore per seed | mean | p99 regret mean |
|---|---|---|---|---|---|
| ideal | full | 4 | -0.064, -0.067, -0.062, -0.065 | -0.065 | 0.018 |
| ideal | nognn | 4 | -0.069, -0.071, -0.066, -0.065 | -0.068 | 0.019 |
| ideal | nomodel | 4 | -0.056, -0.058, -0.052, -0.060 | -0.056 | 0.018 |
| degraded | full | 4 | -0.060, -0.062, -0.060, -0.063 | -0.061 | 0.019 |
| degraded | nognn | 4 | -0.065, -0.065, -0.060, -0.063 | -0.063 | 0.022 |
| degraded | nomodel | 4 | -0.047, -0.047, -0.044, -0.050 | -0.047 | 0.029 |

Paired over graphs (seed-averaged per graph): difference ablation - full model, bootstrap 95% CI over graphs; negative = ablation better
  ideal     nognn    (n_seeds=4 vs 4): -0.0031 [-0.0054, -0.0011]; ablation better on 14/24 graphs
  ideal     nomodel  (n_seeds=4 vs 4): +0.0082 [+0.0048, +0.0121]; ablation better on 3/24 graphs
  degraded  nognn    (n_seeds=4 vs 4): -0.0020 [-0.0040, -0.0002]; ablation better on 14/24 graphs
  degraded  nomodel  (n_seeds=4 vs 4): +0.0144 [+0.0085, +0.0206]; ablation better on 2/24 graphs
