#!/bin/bash
# extra seeds for the two ablations that matter most (no-GNN, no-model-features); 12+4 iterations like the headline models
cd "/x/CN Project/stale-but-safe-te"
C="-m stale_but_safe.train.dfm --workers 3 --max-samples 12000 --eps-per-iter 40 --mode random"
for s in 1 2 3; do
  for x in nognn nomodel; do
    [ -f runs/abl_${x}_final_s$s.pt ] && continue
    flag=$([ $x = nognn ] && echo --no-gnn || echo --no-model)
    python -u $C $flag --seed $s --iters 12 --lr 1e-3 --out runs/abl_${x}_s$s.pt > runs/abl_${x}_s$s.log 2>&1
    python -u $C $flag --seed $s --init runs/abl_${x}_s$s.pt --iters 4 --lr 2e-4 --out runs/abl_${x}_final_s$s.pt > runs/abl_${x}_anneal_s$s.log 2>&1
    echo "$x seed$s $?" >> runs/abl_seeds.status
  done
done
echo "TRAIN DONE" >> runs/abl_seeds.status
