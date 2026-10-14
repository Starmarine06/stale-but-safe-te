Paired window-level comparison, 16 fresh windows per cell (zero-traffic windows skipped), load 0.8. diff = learned - heuristic gain (points);
95% bootstrap CI over windows. Heuristic = tuner seed with best tuning score. Verdict: L = CI above 0, H = CI below 0, = otherwise.

| network | d | P | heuristic % | learned % | diff [95% CI] | verdict | windows |
|---|---|---|---|---|---|---|---|
| abilene | 0 | 1 | 44.9 | 30.9 | -14.0 [-27.8, -5.6] | H | 16 |
| abilene | 0 | 5 | 40.8 | 30.6 | -10.3 [-25.9, -0.3] | H | 16 |
| abilene | 0 | 20 | 38.2 | 32.0 | -6.2 [-21.8, +2.9] | = | 16 |
| abilene | 3 | 1 | 42.6 | 31.2 | -11.4 [-26.8, -1.8] | H | 16 |
| abilene | 3 | 5 | 40.9 | 28.6 | -12.3 [-29.4, +0.4] | = | 16 |
| abilene | 3 | 20 | 35.8 | 29.8 | -6.0 [-22.7, +5.8] | = | 16 |
| abilene | 10 | 1 | 37.0 | 31.8 | -5.1 [-22.4, +5.0] | = | 16 |
| abilene | 10 | 5 | 37.4 | 31.7 | -5.7 [-21.3, +4.0] | = | 16 |
| abilene | 10 | 20 | 32.8 | 29.8 | -3.0 [-18.7, +7.0] | = | 16 |
| geant | 0 | 1 | 38.5 | 34.4 | -4.1 [-5.2, -3.0] | H | 14 |
| geant | 0 | 5 | 33.9 | 34.2 | +0.4 [-1.1, +1.9] | = | 14 |
| geant | 0 | 20 | 22.2 | 32.9 | +10.7 [+8.3, +12.8] | L | 14 |
| geant | 3 | 1 | 33.8 | 34.3 | +0.5 [-1.6, +2.9] | = | 14 |
| geant | 3 | 5 | 31.7 | 34.2 | +2.5 [+1.0, +4.2] | L | 14 |
| geant | 3 | 20 | 20.4 | 32.7 | +12.3 [+9.3, +15.0] | L | 14 |
| geant | 10 | 1 | 29.5 | 33.2 | +3.7 [+1.1, +6.4] | L | 14 |
| geant | 10 | 5 | 23.6 | 32.9 | +9.2 [+7.3, +10.9] | L | 14 |
| geant | 10 | 20 | 20.5 | 31.4 | +10.9 [+7.9, +13.5] | L | 14 |
