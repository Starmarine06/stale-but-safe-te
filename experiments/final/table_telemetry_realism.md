| impairment | heuristic parameters | heuristic gain % | learned gain % (mean of 4 models) | learned - heuristic [95% CI over graphs], points | p99 regret heuristic / learned |
|---|---|---|---|---|---|
| degraded (plain) | tuned for plain degraded | 2.3 | 4.2 | +2.0 [+0.2, +4.2] | +0.331 / +0.075 |
| + 2% gross errors | tuned for plain degraded | 1.6 | 4.2 | +2.6 [+0.3, +5.7] | +0.477 / +0.059 |
| + 5% gross errors | tuned for plain degraded | 1.5 | 4.2 | +2.8 [+0.4, +5.9] | +0.477 / +0.061 |
| + 5% gross errors | re-tuned for this variant (2 seeds) | 0.7 | 4.2 | +3.5 [+0.8, +7.1] | +0.526 / +0.061 |
| + 10% lost poll rounds | tuned for plain degraded | 2.2 | 4.1 | +1.9 [+0.4, +3.7] | +0.261 / +0.075 |
| + poll jitter 0.3 | tuned for plain degraded | 1.8 | 4.2 | +2.4 [+0.4, +5.0] | +0.355 / +0.079 |
| all three (2%, 5% rounds, jitter 0.3) | tuned for plain degraded | 1.5 | 4.2 | +2.7 [+0.6, +5.2] | +0.351 / +0.067 |
| all three (2%, 5% rounds, jitter 0.3) | re-tuned for this variant (2 seeds) | 1.0 | 4.2 | +3.2 [+0.8, +5.9] | +0.386 / +0.067 |
