| policy [telemetry] | scenarios | emu max-util | sim max-util | time-series Pearson | emu loss | flow-mods/min | ctrl loop ms |
|---|---|---|---|---|---|---|---|
| ecmp [ideal] | 4 | 0.866 | 0.865 | 0.98 | 0.0175 | 0.0 | 9 |
| tuned:snmp [snmp] | 4 | 0.798 | 0.786 | 0.93 | 0.0124 | 32.6 | 9 |
| tuned:degraded [degraded] | 4 | 0.808 | 0.814 | 0.84 | 0.0109 | 30.0 | 9 |
| ours_dfm_final [snmp] | 4 | 0.763 | 0.751 | 0.91 | 0.0089 | 270.0 | 12 |
| ours_dfm_final [degraded] | 4 | 0.772 | 0.757 | 0.88 | 0.0080 | 331.9 | 14 |
| guardL2-0.06@ours_dfm_final [degraded] | 4 | 0.794 | 0.769 | 0.86 | 0.0143 | 258.4 | 12 |
| ours_dfm_final [degraded] hyst0.2 | 4 | 0.790 | 0.757 | 0.83 | 0.0082 | 75.8 | 12 |

over 24 runs: Pearson(emu, sim mean max-util) = 0.99; mean |emu - sim| = 0.017; bias (emu - sim) = +0.010
Kendall tau of policy ranking (emu vs sim) per scenario: [np.float64(0.73), np.float64(0.47), np.float64(0.6), np.float64(0.73)]
