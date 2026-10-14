#!/bin/bash
cd "/x/CN Project/stale-but-safe-te"
for sc in fail shift flash hotspot; do
  python -u experiments/final_eval.py --ours runs/ours_dfm_final.pt --ideal runs/ideal_dfm_final.pt --scenario $sc --topos 12 --profiles snmp,degraded --oracle-max-n 0 --seeds 5 --horizon 100 --out experiments/final/stress_$sc.json > experiments/final/stress_$sc.log 2>&1
  echo "$sc $?" >> experiments/final/stress.status
done
