# Reproduction targets. Windows users: run from Git Bash or WSL. Long jobs: see runs/*.sh for the exact queues used.
PY ?= python
W  ?= 4
TRAIN = -m stale_but_safe.train.dfm --workers $(W) --max-samples 12000 --eps-per-iter 40
.PHONY: test data tune train-ours train-ideal eval-zoo eval-real tables audit demo scenario emu-build emu paper-tables

test:            ; $(PY) -m pytest -q
data:            ; $(PY) -m stale_but_safe.env.real prepare germany17 geant germany50 abilene brain   # after downloading SNDlib archives into data/real/
tune:            ; $(PY) -m stale_but_safe.eval.tune
train-ours:      ; $(PY) -u $(TRAIN) --mode random --iters 12 --lr 1e-3 --out runs/ours_dfm_s0.pt && \
                   $(PY) -u $(TRAIN) --mode random --init runs/ours_dfm_s0.pt --iters 4 --lr 2e-4 --out runs/ours_dfm_final.pt
train-ideal:     ; $(PY) -u $(TRAIN) --mode ideal --iters 12 --lr 1e-3 --out runs/ideal_dfm_s0.pt && \
                   $(PY) -u $(TRAIN) --mode ideal --init runs/ideal_dfm_s0.pt --iters 4 --lr 2e-4 --out runs/ideal_dfm_final.pt
eval-zoo:        ; $(PY) experiments/final_eval.py --ours runs/ours_dfm_final.pt --ideal runs/ideal_dfm_final.pt --out experiments/final/results_ours.json
eval-real:       ; for n in abilene germany17 geant germany50 brain; do $(PY) experiments/real_eval.py $$n --out experiments/final/real_$$n.json; done
tables:          ; $(PY) experiments/analyze_final.py experiments/final/results_ours.json experiments/final/results_ideal.json && \
                   $(PY) experiments/analyze_seeds.py && $(PY) experiments/analyze_real.py && $(PY) experiments/analyze_ablations.py && \
                   $(PY) experiments/analyze_ft.py && $(PY) experiments/sensitivity_lossweight.py
audit:           ; $(PY) -m stale_but_safe.cli audit --network abilene --polling 5 --delay 3 --noise 0.2 --loss 0.1 --out reports/abilene
demo:            ; $(PY) -m stale_but_safe.cli demo --topology abilene --mode shadow --model runs/ours_dfm_final.pt
scenario:        ; $(PY) -m stale_but_safe.cli scenario --out runs/emu
emu-build:       ; docker build -f docker/mininet.Dockerfile -t sbs-mininet . && docker build -f docker/ryu.Dockerfile -t sbs-ryu .
emu:             ; $(PY) -m stale_but_safe.emu.run_emulation --scenario runs/emu/scenario.json $(RUNS)
