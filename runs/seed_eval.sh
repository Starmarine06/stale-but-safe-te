#!/bin/bash
cd "/x/CN Project/stale-but-safe-te"
until grep -q "ALL DONE" runs/seeds.status 2>/dev/null; do sleep 60; done
for s in 1 2 3; do
  [ -f experiments/final/results_seed$s.json ] && continue
  python experiments/final_eval.py --ours runs/ours_dfm_final_s$s.pt --ideal runs/ideal_dfm_final_s$s.pt     --only-models --out experiments/final/results_seed$s.json > experiments/final/seed$s.log 2>&1
  echo "eval seed$s $?" >> runs/seed_eval.status
done
echo "EVAL DONE" >> runs/seed_eval.status
