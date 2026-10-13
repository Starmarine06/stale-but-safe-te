#!/bin/bash
cd "/x/CN Project/stale-but-safe-te"
chain() { for n in "$@"; do python -m stale_but_safe.cli audit --network $n --polling 5 --delay 3 --noise 0.2 --loss 0.1 --out reports/$n > runs/audit_$n.log 2>&1; done; }
chain abilene geant &
chain germany50 brain &
wait
echo done > runs/audits3.status
