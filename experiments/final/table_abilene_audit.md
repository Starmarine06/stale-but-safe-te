| telemetry | window set | heuristic params | heuristic % | learned % | learned - heuristic [95% CI over windows] |
|---|---|---|---|---|---|
| snmp | A (20000+, 12 windows) | yaml (per-profile, 40 trials) | 27.9 | 36.3 | +8.4 [+4.2, +12.3] |
| snmp | A (20000+, 12 windows) | grid best-of-5 (d=3,P=5) | 17.7 | 36.3 | +18.6 [+5.7, +32.0] |
| snmp | B (15000+, 16 windows) | yaml (per-profile, 40 trials) | 32.7 | 31.4 | -1.3 [-18.5, +9.9] |
| snmp | B (15000+, 16 windows) | grid best-of-5 (d=3,P=5) | 29.3 | 31.4 | +2.1 [-16.0, +15.0] |
| degraded | A (20000+, 12 windows) | yaml (per-profile, 40 trials) | 17.0 | 36.1 | +19.1 [+9.0, +30.2] |
| degraded | A (20000+, 12 windows) | grid best-of-5 (d=3,P=5) | 31.2 | 36.1 | +4.9 [+1.3, +9.0] |
| degraded | B (15000+, 16 windows) | yaml (per-profile, 40 trials) | 24.2 | 28.6 | +4.5 [-14.3, +19.0] |
| degraded | B (15000+, 16 windows) | grid best-of-5 (d=3,P=5) | 35.5 | 28.6 | -6.8 [-23.6, +5.6] |
| clean d=3,P=5 | A (20000+, 12 windows) | grid best-of-5 (d=3,P=5) | 37.0 | 36.6 | -0.4 [-2.5, +1.8] |
| clean d=3,P=5 | B (15000+, 16 windows) | grid best-of-5 (d=3,P=5) | 40.9 | 28.6 | -12.3 [-31.9, +0.3] |
