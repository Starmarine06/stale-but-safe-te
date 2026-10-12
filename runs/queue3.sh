#!/bin/bash
cd "/x/CN Project/stale-but-safe-te"
C="-m stale_but_safe.train.dfm --workers 5 --max-samples 12000 --eps-per-iter 40 --mode random"
for v in "noh:--H 1" "noage:--no-age" "nognn:--no-gnn"; do
  name=${v%%:*}; flags=${v#*:}
  python -u $C $flags --iters 12 --lr 1e-3 --out runs/abl_${name}_s0.pt > runs/abl_${name}_s0.log 2>&1
  python -u $C --init runs/abl_${name}_s0.pt --iters 4 --lr 2e-4 --out runs/abl_${name}_final.pt > runs/abl_${name}_anneal.log 2>&1
  echo "$name $?" >> runs/queue3.status
done
