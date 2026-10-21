#!/bin/bash
# TCP batch 2 (design: docs/tcp_batch2_design.md): ecmp, tuned (degraded), learned (degraded); rep 0, then reps 1 and 2
cd "/x/CN Project/stale-but-safe-te"
ST=runs/emu_tcp_b_status.txt
NAMES="b01 b02 b03 b04 b05 b06 b07 b08 b09 b10 b11 b12 b13 b14 b15 b16 b17 b18"
for rep in 0 1 2; do
  if [ $rep = 0 ]; then root=runs/emu_tcp_b; else root=runs/emu_tcp_b_r$rep; fi
  for n in $NAMES; do
    M=$(cat runs/emu_tcp_b/$n/model.txt)
    mkdir -p $root/$n; cp runs/emu_tcp_b/$n/scenario.json $root/$n/; cp runs/emu_tcp_b/$n/model.txt $root/$n/
    python -m stale_but_safe.emu.run_emulation --scenario $root/$n/scenario.json --out $root/$n --proto tcp \
      --run ecmp:ideal --run tunedv2:degraded:degraded --run model:$M:degraded >> runs/emu_tcp_b_log.txt 2>&1
    echo "rep$rep $n done" >> $ST
  done
done
echo "TCPB DONE" >> $ST
