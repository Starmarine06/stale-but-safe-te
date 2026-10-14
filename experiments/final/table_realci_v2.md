| network | load | telemetry | tuned gain % | learned gain % | learned - tuned, pts (95% paired CI over windows) | learned better in windows | verdict |
|---|---|---|---|---|---|---|---|
| abilene | 0.6 | ideal | 38.6 | 33.9 | -4.6 [-6.4, -2.8] | 2/12 | tuned |
| abilene | 0.6 | rich | 37.3 | 34.0 | -3.3 [-5.2, -1.4] | 3/12 | tuned |
| abilene | 0.6 | snmp | 32.0 | 34.8 | +2.8 [+1.4, +4.4] | 9/12 | learned |
| abilene | 0.6 | degraded | 26.9 | 34.7 | +7.8 [+3.0, +13.4] | 9/12 | learned |
| abilene | 0.8 | ideal | 41.8 | 36.1 | -5.7 [-6.9, -4.4] | 0/12 | tuned |
| abilene | 0.8 | rich | 40.1 | 35.8 | -4.3 [-6.1, -2.3] | 2/12 | tuned |
| abilene | 0.8 | snmp | 37.0 | 36.3 | -0.8 [-2.9, +1.3] | 5/12 | tie |
| abilene | 0.8 | degraded | 26.5 | 36.1 | +9.6 [+4.3, +15.9] | 11/12 | learned |
| abilene | 1.0 | ideal | 43.7 | 37.2 | -6.6 [-7.8, -5.2] | 0/12 | tuned |
| abilene | 1.0 | rich | 41.4 | 36.7 | -4.7 [-6.0, -2.9] | 2/12 | tuned |
| abilene | 1.0 | snmp | 40.2 | 36.8 | -3.5 [-5.4, -1.3] | 1/12 | tuned |
| abilene | 1.0 | degraded | 25.7 | 36.5 | +10.8 [+5.0, +17.6] | 10/12 | learned |
| geant | 0.6 | ideal | 43.4 | 41.0 | -2.4 [-3.8, -1.1] | 1/12 | tuned |
| geant | 0.6 | rich | 40.3 | 41.2 | +0.9 [-0.6, +2.2] | 7/12 | tie |
| geant | 0.6 | snmp | 37.2 | 41.7 | +4.6 [+2.4, +6.6] | 10/12 | learned |
| geant | 0.6 | degraded | 28.9 | 41.7 | +12.7 [+9.5, +16.0] | 12/12 | learned |
| geant | 0.8 | ideal | 46.8 | 43.7 | -3.0 [-4.7, -1.4] | 2/12 | tuned |
| geant | 0.8 | rich | 43.6 | 43.7 | +0.1 [-1.7, +2.0] | 7/12 | tie |
| geant | 0.8 | snmp | 41.9 | 43.9 | +2.0 [-0.2, +4.1] | 9/12 | tie |
| geant | 0.8 | degraded | 30.9 | 43.6 | +12.6 [+9.3, +15.9] | 12/12 | learned |
| geant | 1.0 | ideal | 48.7 | 44.9 | -3.8 [-5.5, -2.0] | 1/12 | tuned |
| geant | 1.0 | rich | 45.7 | 44.7 | -1.0 [-3.0, +1.1] | 6/12 | tie |
| geant | 1.0 | snmp | 44.2 | 44.7 | +0.5 [-1.5, +2.6] | 6/12 | tie |
| geant | 1.0 | degraded | 32.2 | 44.1 | +12.0 [+8.8, +15.1] | 12/12 | learned |
| germany17 | 0.6 | ideal | 19.3 | 14.0 | -5.3 [-5.9, -4.7] | 0/12 | tuned |
| germany17 | 0.6 | rich | 14.7 | 13.8 | -0.9 [-1.6, -0.2] | 1/12 | tuned |
| germany17 | 0.6 | snmp | 11.6 | 13.9 | +2.3 [+1.3, +3.3] | 10/12 | learned |
| germany17 | 0.6 | degraded | 2.8 | 12.3 | +9.6 [+7.6, +11.6] | 12/12 | learned |
| germany17 | 0.8 | ideal | 22.7 | 15.7 | -7.0 [-8.1, -6.0] | 0/12 | tuned |
| germany17 | 0.8 | rich | 17.7 | 15.4 | -2.3 [-3.0, -1.7] | 0/12 | tuned |
| germany17 | 0.8 | snmp | 15.0 | 15.3 | +0.3 [-0.6, +1.1] | 6/12 | tie |
| germany17 | 0.8 | degraded | 2.1 | 13.7 | +11.6 [+9.4, +13.8] | 12/12 | learned |
| germany17 | 1.0 | ideal | 24.2 | 16.5 | -7.7 [-8.7, -6.8] | 0/12 | tuned |
| germany17 | 1.0 | rich | 19.3 | 16.0 | -3.3 [-4.2, -2.3] | 1/12 | tuned |
| germany17 | 1.0 | snmp | 17.8 | 15.9 | -1.9 [-3.1, -0.7] | 4/12 | tuned |
| germany17 | 1.0 | degraded | 1.5 | 14.4 | +13.0 [+10.6, +15.1] | 12/12 | learned |

Cells where the paired 95% interval excludes 0: {'learned': 12, 'tuned': 17, 'tie': 7}
  ideal: {'learned': 0, 'tuned': 9, 'tie': 0}
  rich: {'learned': 0, 'tuned': 6, 'tie': 3}
  snmp: {'learned': 3, 'tuned': 2, 'tie': 4}
  degraded: {'learned': 9, 'tuned': 0, 'tie': 0}
