| case | guard | fallback | false rejection | harmful passed | harm caught | gain kept | p99 regret raw -> guarded |
|---|---|---|---|---|---|---|---|
| Zoo (10 graphs, n<=25) ideal | L2-only eps=-0.06 (headline) | 71% | 12% | 4.5% | 43% | 83% | +0.050 -> +0.019 |
| Zoo (10 graphs, n<=25) ideal | L2-only eps=0 | 32% | 7% | 2.4% | 26% | 96% | +0.050 -> +0.035 |
| Zoo (10 graphs, n<=25) ideal | all layers eps=0, tau=0.05 (default) | 32% | 7% | 2.4% | 26% | 96% | +0.050 -> +0.035 |
| Zoo (10 graphs, n<=25) degraded | L2-only eps=-0.06 (headline) | 72% | 14% | 7.4% | 22% | 76% | +0.065 -> +0.057 |
| Zoo (10 graphs, n<=25) degraded | L2-only eps=0 | 31% | 8% | 3.8% | 1% | 95% | +0.063 -> +0.063 |
| Zoo (10 graphs, n<=25) degraded | all layers eps=0, tau=0.05 (default) | 67% | 4% | 3.6% | 12% | 84% | +0.023 -> +0.013 |
| Zoo link failure degraded | L2-only eps=-0.06 (headline) | 80% | 13% | 6.4% | 90% | 91% | +0.148 -> +0.021 |
| Zoo link failure degraded | L2-only eps=0 | 49% | 10% | 3.3% | 87% | 109% | +0.148 -> +0.031 |
| Zoo link failure degraded | all layers eps=0, tau=0.05 (default) | 70% | 8% | 2.3% | 94% | 109% | +0.141 -> +0.000 |
| Zoo demand shift degraded | L2-only eps=-0.06 (headline) | 73% | 18% | 7.5% | 53% | 74% | +0.084 -> +0.058 |
| Zoo demand shift degraded | L2-only eps=0 | 30% | 12% | 5.2% | 14% | 93% | +0.072 -> +0.068 |
| Zoo demand shift degraded | all layers eps=0, tau=0.05 (default) | 70% | 4% | 3.7% | 33% | 84% | +0.030 -> +0.009 |
| abilene real trace ideal | L2-only eps=-0.06 (headline) | 0% | nan% | 0.0% | nan% | 100% | -0.060 -> -0.060 |
| abilene real trace ideal | L2-only eps=0 | 0% | nan% | 0.0% | nan% | 100% | -0.060 -> -0.060 |
| abilene real trace ideal | all layers eps=0, tau=0.05 (default) | 0% | nan% | 0.0% | nan% | 100% | -0.060 -> -0.060 |
| abilene real trace degraded | L2-only eps=-0.06 (headline) | 1% | 0% | 0.0% | nan% | 100% | +0.000 -> +0.000 |
| abilene real trace degraded | L2-only eps=0 | 1% | 0% | 0.0% | nan% | 100% | +0.000 -> +0.000 |
| abilene real trace degraded | all layers eps=0, tau=0.05 (default) | 3% | 11% | 0.0% | nan% | 100% | +0.000 -> +0.000 |
| geant real trace ideal | L2-only eps=-0.06 (headline) | 1% | 100% | 0.0% | nan% | 100% | -0.030 -> -0.020 |
| geant real trace ideal | L2-only eps=0 | 0% | nan% | 0.0% | nan% | 100% | -0.030 -> -0.030 |
| geant real trace ideal | all layers eps=0, tau=0.05 (default) | 0% | nan% | 0.0% | nan% | 100% | -0.030 -> -0.030 |
| geant real trace degraded | L2-only eps=-0.06 (headline) | 2% | 71% | 0.0% | nan% | 100% | -0.006 -> +0.000 |
| geant real trace degraded | L2-only eps=0 | 0% | 0% | 0.0% | nan% | 100% | -0.008 -> -0.008 |
| geant real trace degraded | all layers eps=0, tau=0.05 (default) | 12% | 11% | 0.0% | nan% | 100% | +0.000 -> +0.000 |
