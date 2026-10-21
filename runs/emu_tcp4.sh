#!/bin/bash
# guard0.05 and guardL2-0.06 with controller hysteresis 0.2 (rep 0 scenarios), to test whether hysteresis rescues the guard
cd "/x/CN Project/stale-but-safe-te"
ST=runs/emu_tcp4_status.txt
run() { out=$1; name=$2; shift 2
  mkdir -p $out/$name; cp runs/emu_tcp/$name/scenario.json $out/$name/; cp runs/emu_tcp/$name/model.txt $out/$name/ 2>/dev/null
  python -m stale_but_safe.emu.run_emulation --scenario $out/$name/scenario.json --out $out/$name --proto tcp "$@" >> runs/emu_tcp4_log.txt 2>&1; }
modelof() { n=$1; if [ -f runs/emu_tcp/$n/model.txt ]; then cat runs/emu_tcp/$n/model.txt; else echo runs/ours_dfm_final_$n.pt; fi; }
for n in s1 s2 s3 n14 n15 n16 n17 n18 n19; do
  M=$(modelof $n)
  run runs/emu_tcp_gh $n --hyst 0.2 --run guard0.05@model:$M:degraded --run guardL2-0.06@model:$M:degraded
  echo "gh $n done" >> $ST
done
echo "TCP4 DONE" >> $ST
