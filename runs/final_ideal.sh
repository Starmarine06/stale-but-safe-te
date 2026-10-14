#!/bin/bash
cd "/x/CN Project/stale-but-safe-te"
python -u experiments/final_eval.py --ideal runs/ideal_dfm_final.pt --only-models --seeds 5 --horizon 100 --out experiments/final/results_ideal.json > experiments/final/ideal.log 2>&1
echo "exit $?" >> experiments/final/ideal.log
