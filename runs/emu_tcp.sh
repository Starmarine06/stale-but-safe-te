#!/bin/bash
cd "/x/CN Project/stale-but-safe-te"
for s in s1 s2 s3; do
  python -m stale_but_safe.emu.run_emulation --scenario runs/emu_tcp/$s/scenario.json --out runs/emu_tcp/$s --proto tcp \
    --run ecmp:ideal --run tunedv2:degraded:degraded --run model:runs/ours_dfm_final_$s.pt:degraded \
    --run guard0.05@model:runs/ours_dfm_final_$s.pt:degraded >> runs/emu_tcp/log.txt 2>&1
  echo "$s done" >> runs/emu_tcp/status.txt
done
echo "EMU TCP DONE" >> runs/emu_tcp/status.txt
