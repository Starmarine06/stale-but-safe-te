#!/bin/bash
# extra training seeds for the headline configs (skips finished ones, so it can be re-run after a crash)
cd "/x/CN Project/stale-but-safe-te"
C="-m stale_but_safe.train.dfm --workers 4 --max-samples 12000 --eps-per-iter 40"
for s in 1 2 3; do
  for mode in random ideal; do
    tag=$([ $mode = random ] && echo ours || echo ideal)
    [ -f runs/${tag}_dfm_final_s$s.pt ] && continue
    python -u $C --mode $mode --seed $s --iters 12 --lr 1e-3 --out runs/${tag}_dfm_s$s.pt > runs/${tag}_dfm_s$s.log 2>&1
    python -u $C --mode $mode --seed $s --init runs/${tag}_dfm_s$s.pt --iters 4 --lr 2e-4 --out runs/${tag}_dfm_final_s$s.pt > runs/${tag}_dfm_anneal_s$s.log 2>&1
    echo "$tag seed$s $? $(date +%H:%M)" >> runs/seeds.status
  done
done
echo "ALL DONE" >> runs/seeds.status
