#!/bin/bash
cd "/x/CN Project/stale-but-safe-te"
C="-m stale_but_safe.train.dfm --workers 4 --max-samples 12000 --eps-per-iter 40"
python -u $C --mode random --init runs/dfm_ckpt12.pt --iters 4 --lr 2e-4 --out runs/ours_dfm_final.pt > runs/ours_dfm_anneal.log 2>&1
echo "ours done $?" >> runs/queue1.status
python -u $C --mode ideal --iters 12 --lr 1e-3 --out runs/ideal_dfm_s0.pt > runs/ideal_dfm_s0.log 2>&1
python -u $C --mode ideal --init runs/ideal_dfm_s0.pt --iters 4 --lr 2e-4 --out runs/ideal_dfm_final.pt > runs/ideal_dfm_anneal.log 2>&1
echo "ideal done $?" >> runs/queue1.status
