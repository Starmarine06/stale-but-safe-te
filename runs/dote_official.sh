#!/bin/bash
cd "$(dirname "$0")/.."
for n in abilene geant; do
  python experiments/dote_official_bench.py $n --out experiments/final/dote_official_$n.json > experiments/final/dote_official_$n.log 2>&1
done
echo DONE > runs/dote_official_status.txt
