#!/bin/bash
cd "X:/CN Project/stale-but-safe-te"
python -m stale_but_safe.emu.run_emulation --scenario runs/emu/scenario.json \
 --run ecmp:ideal --run tuned:snmp:snmp --run tuned:degraded:degraded \
 --run model:runs/ours_dfm_final.pt:snmp --run model:runs/ours_dfm_final.pt:degraded \
 --run model:runs/ideal_dfm_final.pt:degraded \
 --run guard0.05@model:runs/ours_dfm_final.pt:degraded > runs/emu_all.log 2>&1
echo "exit $?" > runs/emu_all.status
