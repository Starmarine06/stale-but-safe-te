#!/bin/bash
# second, larger set of independent trace windows (10 per cell) to check the 4-window results
cd "/x/CN Project/stale-but-safe-te"
for n in abilene germany17 geant germany50 brain; do
  python experiments/real_eval.py $n --seeds 10 --seed0 7000 --out experiments/final/realw_$n.json > experiments/final/realw_$n.log 2>&1 &
  [ $(jobs -r | wc -l) -ge 3 ] && wait -n
done
wait
echo "REALW DONE" > runs/realw_eval.status
