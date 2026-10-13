| policy | telemetry | emu max-util | sim max-util | Pearson | Spearman | emu loss | sim loss | ctrl latency ms | flow-mods/min | fallback |
|---|---|---|---|---|---|---|---|---|---|---|
| ecmp | ideal | 0.925 | 0.921 | 0.96 | 0.85 | 0.0334 | 0.0364 | 18 | 0.0 | - |
| tuned:snmp | snmp | 0.875 | 0.873 | 0.98 | 0.89 | 0.0219 | 0.0255 | 17 | 25.5 | - |
| tuned:degraded | degraded | 0.894 | 0.914 | 0.93 | 0.76 | 0.0203 | 0.0283 | 16 | 27.0 | - |
| ours_dfm_final | snmp | 0.901 | 0.881 | 0.93 | 0.86 | 0.0202 | 0.0236 | 32 | 265.5 | - |
| ours_dfm_final | degraded | 0.880 | 0.889 | 0.95 | 0.90 | 0.0179 | 0.0250 | 26 | 330.0 | - |
| ideal_dfm_final | degraded | 0.885 | 0.890 | 0.94 | 0.91 | 0.0201 | 0.0263 | 28 | 408.0 | - |
| guard0.05@ours_dfm_final | degraded | 0.910 | 0.891 | 0.90 | 0.84 | 0.0240 | 0.0263 | 27 | 376.5 | 16/39 |

across 7 runs: Pearson(emu, sim mean max-util) = 0.66; Kendall tau of method ranking = 0.62; mean |emu-sim| = 0.011
