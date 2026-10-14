#!/bin/bash
cd "/x/CN Project/stale-but-safe-te"
for n in abilene geant germany50 brain; do
  python -m stale_but_safe.cli audit --network $n --polling 5 --delay 3 --noise 0.2 --loss 0.1 --out reports/$n > runs/audit_$n.log 2>&1
done
echo done > runs/audits.status
