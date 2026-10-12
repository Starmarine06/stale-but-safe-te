#!/bin/bash
cd "/x/CN Project/stale-but-safe-te"
while [ "$(cat runs/queue3.status 2>/dev/null | wc -l)" -lt 3 ]; do sleep 30; done
C="-m stale_but_safe.train.dfm --workers 5 --max-samples 12000 --eps-per-iter 40 --mode random"
python -u $C --no-model --iters 12 --lr 1e-3 --out runs/abl_nomodel_s0.pt > runs/abl_nomodel_s0.log 2>&1
python -u $C --init runs/abl_nomodel_s0.pt --iters 4 --lr 2e-4 --out runs/abl_nomodel_final.pt > runs/abl_nomodel_anneal.log 2>&1
echo "nomodel $?" >> runs/queue4.status
