#!/bin/bash
cd "/x/CN Project/stale-but-safe-te"
ST=runs/emu_tcp3_status.txt
run() { # outdir name runs...
  out=$1; name=$2; shift 2
  mkdir -p $out/$name; cp runs/emu_tcp/$name/scenario.json $out/$name/; cp runs/emu_tcp/$name/model.txt $out/$name/ 2>/dev/null
  python -m stale_but_safe.emu.run_emulation --scenario $out/$name/scenario.json --out $out/$name --proto tcp "$@" >> runs/emu_tcp3_log.txt 2>&1
}
modelof() { n=$1; if [ -f runs/emu_tcp/$n/model.txt ]; then cat runs/emu_tcp/$n/model.txt; else echo runs/ours_dfm_final_$n.pt; fi; }
NAMES="s1 s2 s3 n14 n15 n16 n17 n18 n19"
for rep in 1 2; do
  for n in $NAMES; do
    M=$(modelof $n)
    run runs/emu_tcp_r$rep $n --run ecmp:ideal --run tunedv2:degraded:degraded --run model:$M:degraded --run guard0.05@model:$M:degraded
    echo "rep$rep $n done" >> $ST
  done
done
for n in $NAMES; do
  M=$(modelof $n)
  run runs/emu_tcp_var $n --run guardL2-0.06@model:$M:degraded
  run runs/emu_tcp_varh $n --hyst 0.2 --run model:$M:degraded
  echo "var $n done" >> $ST
done
echo "TCP3 DONE" >> $ST
