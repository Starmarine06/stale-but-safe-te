#!/bin/bash
cd "/x/CN Project/stale-but-safe-te"
mkdir -p runs/emu3
for name in emu10_s0 emu10_s1 emu10_s2 er12_s0; do
  mkdir -p runs/emu3/$name; cp runs/emu2/$name/scenario.json runs/emu3/$name/
  python -m stale_but_safe.emu.run_emulation --scenario runs/emu3/$name/scenario.json --out runs/emu3/$name \
    --run tunedv2:snmp:snmp --run tunedv2:degraded:degraded >> runs/emu3/log.txt 2>&1
  echo "$name done" >> runs/emu3/status.txt
done
echo "EMU DONE" >> runs/emu3/status.txt
