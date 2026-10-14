#!/bin/bash
cd "/x/CN Project/stale-but-safe-te"
until grep -q "EVAL DONE" runs/seed_eval.status 2>/dev/null; do sleep 30; done
python experiments/final_eval.py --ours runs/ours_dfm_final.pt --only-models --include-tuned-guard --eps -0.06 --guard-layers 2 \
  --out experiments/final/results_guardL2.json > experiments/final/guardL2.log 2>&1
echo "guardL2 $?" >> runs/ablate.status
for x in noh noage nognn nomodel; do
  python experiments/final_eval.py --ours runs/abl_${x}_final.pt --only-models --profiles ideal,degraded \
    --out experiments/final/results_abl_$x.json > experiments/final/abl_$x.log 2>&1
  echo "$x $?" >> runs/ablate.status
done
echo "ABLATE DONE" >> runs/ablate.status
