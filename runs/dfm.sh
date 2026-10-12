#!/bin/bash
cd "/x/CN Project/stale-but-safe-te"
python -u -m stale_but_safe.train.dfm --mode random --iters 40 --eps-per-iter 40 --workers 4 --max-samples 12000 --out runs/ours_dfm_s0.pt > runs/ours_dfm_s0.log 2>&1
echo "exit $?" >> runs/ours_dfm_s0.log
