#!/bin/bash
cd "/x/CN Project/stale-but-safe-te"
python -u experiments/final_eval.py --ours runs/ours_dfm_final.pt --seeds 5 --horizon 100 --out experiments/final/results_ours.json > experiments/final/ours.log 2>&1
echo "exit $?" >> experiments/final/ours.log
