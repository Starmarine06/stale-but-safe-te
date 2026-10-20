#!/bin/bash
cd "/x/CN Project/stale-but-safe-te"
for name in n14 n16 n18; do
  M=$(cat runs/emu_tcp/$name/model.txt)
  mkdir -p runs/emu_shadow/$name; cp runs/emu_tcp/$name/scenario.json runs/emu_shadow/$name/
  python -m stale_but_safe.emu.run_emulation --scenario runs/emu_shadow/$name/scenario.json --out runs/emu_shadow/$name --proto tcp --shadow \
    --run model:$M:degraded >> runs/emu_shadow/log.txt 2>&1
  echo "$name done" >> runs/emu_shadow/status.txt
done
echo "SHADOW DONE" >> runs/emu_shadow/status.txt
