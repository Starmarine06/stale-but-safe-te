Emulation, 4 scenarios (emu10 x3 seeds, er12), mean over scenarios

| policy [telemetry] | emu max-util | sim max-util | emu loss | flow-mods/min | ctrl loop ms |
|---|---|---|---|---|---|
| original heuristic [snmp] | 0.798 | 0.786 | 0.0124 | 32.6 | 9 |
| re-tuned heuristic [snmp] | 0.772 | 0.760 | 0.0095 | 139.5 | 14 |
| learned [snmp] | 0.763 | 0.751 | 0.0089 | 270.0 | 12 |
| original heuristic [degraded] | 0.808 | 0.814 | 0.0109 | 30.0 | 9 |
| re-tuned heuristic [degraded] | 0.806 | 0.795 | 0.0095 | 139.1 | 14 |
| learned [degraded] | 0.772 | 0.757 | 0.0080 | 331.9 | 14 |
| ECMP | 0.866 | 0.865 | 0.0175 | 0.0 | 9 |

re-tuned heuristic runs (n=8): Pearson(emu, sim mean max-util) = 1.00; mean |emu - sim| = 0.018

Per scenario, emulated mean max-util (lower = better):

| scenario | ECMP | re-tuned snmp | learned snmp | re-tuned degraded | learned degraded |
|---|---|---|---|---|---|
| emu10_s0 | 0.911 | 0.883 | 0.880 | 0.896 | 0.895 |
| emu10_s1 | 0.806 | 0.818 | 0.780 | 0.825 | 0.764 |
| emu10_s2 | 0.894 | 0.871 | 0.832 | 0.904 | 0.865 |
| er12_s0 | 0.852 | 0.518 | 0.559 | 0.599 | 0.562 |
