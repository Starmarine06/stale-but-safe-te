#!/bin/bash
cd "/x/CN Project/stale-but-safe-te"
(python -u experiments/final_eval.py --tuned-only --tuned-config configs/tuned_heuristic_v2.yaml --include-tuned-guard --eps -0.06 --guard-layers 2 --out experiments/final/results_v2_zoo.json > experiments/final/v2_zoo.log 2>&1; echo "zoo $?" >> runs/rerun_v2.status) &
(for sc in fail shift flash hotspot; do python -u experiments/final_eval.py --tuned-only --tuned-config configs/tuned_heuristic_v2.yaml --scenario $sc --topos 12 --profiles snmp,degraded --seeds 5 --horizon 100 --out experiments/final/stress_v2_$sc.json > experiments/final/stress_v2_$sc.log 2>&1; echo "stress $sc $?" >> runs/rerun_v2.status; done) &
(for n in abilene geant germany17; do python -u experiments/realci_v2.py $n > experiments/final/realci_v2_$n.log 2>&1; echo "real $n $?" >> runs/rerun_v2.status; done) &
wait
echo ALLDONE >> runs/rerun_v2.status
