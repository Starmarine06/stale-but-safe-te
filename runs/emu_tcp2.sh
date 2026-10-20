#!/bin/bash
cd "/x/CN Project/stale-but-safe-te"
for name in n14 n15 n16 n17 n18 n19; do
  M=$(cat runs/emu_tcp/$name/model.txt)
  python -m stale_but_safe.emu.run_emulation --scenario runs/emu_tcp/$name/scenario.json --out runs/emu_tcp/$name --proto tcp \
    --run ecmp:ideal --run tunedv2:degraded:degraded --run model:$M:degraded --run guard0.05@model:$M:degraded >> runs/emu_tcp/log2.txt 2>&1
  echo "$name done" >> runs/emu_tcp/status2.txt
done
echo "EMU TCP2 DONE" >> runs/emu_tcp/status2.txt
