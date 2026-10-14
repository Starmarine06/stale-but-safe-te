#!/bin/bash
cd "/x/CN Project/stale-but-safe-te"
for n in abilene germany17 geant germany50 brain; do
  python experiments/real_eval.py $n --out experiments/final/real_$n.json > experiments/final/real_$n.log 2>&1 &
  [ $(jobs -r | wc -l) -ge 3 ] && wait -n
done
wait
echo "REAL DONE" > runs/real_eval.status
