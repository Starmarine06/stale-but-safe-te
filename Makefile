# Reproduction targets. Windows users: run from Git Bash or WSL.
PY ?= python
.PHONY: test tune train-ours train-ideal val demo scenario emu-build emu reproduce-all

test:            ; $(PY) -m pytest -q
tune:            ; $(PY) -m stale_but_safe.eval.tune
train-ours:      ; $(PY) -m stale_but_safe.train.bc --mode random --iters 10 --eps-per-iter 40 --workers 5 --max-samples 10000 --out runs/ours_v2_s0.pt
train-ideal:     ; $(PY) -m stale_but_safe.train.bc --mode ideal  --iters 10 --eps-per-iter 40 --workers 5 --max-samples 10000 --out runs/ideal_v2_s0.pt
val:             ; $(PY) experiments/val_compare.py runs/ideal_v2_s0.pt runs/ours_v2_s0.pt 0.05
demo:            ; $(PY) -m stale_but_safe.cli demo --topology abilene --mode shadow
scenario:        ; $(PY) -m stale_but_safe.cli scenario --out runs/emu
emu-build:       ; docker build -f docker/mininet.Dockerfile -t sbs-mininet . && docker build -f docker/ryu.Dockerfile -t sbs-ryu .
emu:             ; $(PY) -m stale_but_safe.emu.run_emulation --scenario runs/emu/scenario.json $(RUNS)
