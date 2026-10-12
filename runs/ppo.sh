#!/bin/bash
cd "/x/CN Project/stale-but-safe-te"
python -u -m stale_but_safe.train.ppo --init runs/ours_v2_s0.pt --iters 30 --eps-per-iter 40 --workers 5 --out runs/ours_ppo_s0.pt > runs/ours_ppo_s0.log 2>&1
echo "exit $?" >> runs/ours_ppo_s0.log
