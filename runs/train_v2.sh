#!/bin/bash
cd "/x/CN Project/stale-but-safe-te"
python -m stale_but_safe.train.bc --mode random --iters 10 --eps-per-iter 40 --epochs 4 --workers 5 --max-samples 10000 --seed 0 --out runs/ours_v2_s0.pt > runs/ours_v2_s0.log 2>&1
python -m stale_but_safe.train.bc --mode ideal --iters 10 --eps-per-iter 40 --epochs 4 --workers 5 --max-samples 10000 --seed 0 --out runs/ideal_v2_s0.pt > runs/ideal_v2_s0.log 2>&1
