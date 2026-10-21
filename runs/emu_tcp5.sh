#!/bin/bash
# instrumented guard runs (guard_trace logged) on the two scenarios where the guard cost most
cd "/x/CN Project/stale-but-safe-te"
ST=runs/emu_tcp5_status.txt
modelof() { n=$1; if [ -f runs/emu_tcp/$n/model.txt ]; then cat runs/emu_tcp/$n/model.txt; else echo runs/ours_dfm_final_$n.pt; fi; }
for n in s2 n16; do
  M=$(modelof $n); out=runs/emu_tcp_diag; mkdir -p $out/$n; cp runs/emu_tcp/$n/scenario.json $out/$n/
  python -m stale_but_safe.emu.run_emulation --scenario $out/$n/scenario.json --out $out/$n --proto tcp \
    --run ecmp:ideal --run model:$M:degraded --run guard0.05@model:$M:degraded --run guardL2-0.06@model:$M:degraded >> runs/emu_tcp5_log.txt 2>&1
  echo "diag $n done" >> $ST
done
echo "TCP5 DONE" >> $ST
