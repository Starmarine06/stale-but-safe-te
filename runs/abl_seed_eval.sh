#!/bin/bash
cd "/x/CN Project/stale-but-safe-te"
for s in 1 2 3; do
  for x in nognn nomodel; do
    while [ ! -f runs/abl_${x}_final_s$s.pt ]; do sleep 30; done
    sleep 5
    python experiments/final_eval.py --ours runs/abl_${x}_final_s$s.pt --only-models --profiles ideal,degraded \
      --out experiments/final/results_abl_${x}_s$s.json > experiments/final/abl_${x}_s$s.log 2>&1
    echo "eval $x seed$s $?" >> runs/abl_seed_eval.status
  done
done
echo "EVAL DONE" >> runs/abl_seed_eval.status
