#!/bin/bash
cd "/x/CN Project/stale-but-safe-te"
M=runs/ours_dfm_final.pt
i=0
for sc in "emu10 0" "emu10 1" "emu10 2" "er12 0"; do
  set -- $sc; topo=$1; seed=$2; name=${topo}_s${seed}
  python -m stale_but_safe.cli scenario --out runs/emu2/$name --topology $topo --load 0.7 --steps 40 --active 24 --seed $seed > /dev/null
  python -m stale_but_safe.emu.run_emulation --scenario runs/emu2/$name/scenario.json --out runs/emu2/$name \
    --run ecmp:ideal --run tuned:snmp:snmp --run tuned:degraded:degraded \
    --run model:$M:snmp --run model:$M:degraded --run guardL2-0.06@model:$M:degraded >> runs/emu2/log.txt 2>&1
  python -m stale_but_safe.emu.run_emulation --scenario runs/emu2/$name/scenario.json --out runs/emu2/${name}_h0.2 --hyst 0.2 \
    --run model:$M:degraded >> runs/emu2/log.txt 2>&1
  cp runs/emu2/$name/scenario.json runs/emu2/${name}_h0.2/ 2>/dev/null
  echo "$name done" >> runs/emu2/status.txt
done
echo "EMU DONE" >> runs/emu2/status.txt
