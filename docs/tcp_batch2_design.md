# TCP emulation batch 2: pre-declared design (written before any batch-2 run)

Purpose: the nine batch-1 TCP scenarios could not resolve the learned-vs-tuned difference. A fluid-simulator replay of the same nine
scenarios predicts a paired max-utilisation difference (learned - tuned) of -0.0063 [-0.0146, +0.0010] (7 of 9 lower, Wilcoxon p = 0.13),
and the emulation measured -0.0065 [-0.0194, +0.0033]. The two agree, and both intervals are wider than the effect. Detecting -0.0063
with paired SD about 0.012 needs roughly 28 scenarios, so batch 2 adds 18 for a total of 27.

Design (fixed before running; all scenarios are run and all are reported, none are dropped or replaced):

| id | topology | traffic pattern | load |
|---|---|---|---|
| b01-b03 | BA16 (topology seeds 5, 6, 7) | gravity + bursts | 0.8 |
| b04-b06 | BA16 (5, 6, 7) | hotspots (hotspot_prob 0.1) | 0.8 |
| b07-b09 | GEO14 (topology seeds 5, 6, 7) | demand shift at interval 45 | 0.8 |
| b10-b12 | GEO14 (5, 6, 7) | flash crowd (at 30, length 20) | 0.8 |
| b13-b14 | BA16 (8), GEO14 (8) | gravity + bursts | 0.6 |
| b15-b16 | BA16 (9), GEO14 (9) | gravity + bursts | 1.0 |
| b17-b18 | Abilene | hotspots, flash crowd | 0.8 |

Common: 60 active demands, 90 one-second intervals, mbps_per_unit 4, TCP (16 connections per demand), degraded telemetry for the
learned and tuned policies, ECMP with ideal telemetry. Learned model = one of the three trained seeds, rotated (s1, s2, s3, ...).
Tuned heuristic = configs/tuned_heuristic_v2.yaml, degraded profile. Scenario seeds 100 + index.
Replicates: each scenario is run once first (rep 0); if time allows a second and third run follow, in the same order.

Primary outcome (declared in advance): paired max-utilisation difference learned - tuned over all 27 scenarios (scenario means),
bootstrap 95% interval and Wilcoxon. Secondary: goodput shortfall difference; per-scenario agreement between the simulator replay and the
emulation for the learned - tuned difference (Pearson r); results split by traffic pattern (reported as descriptive only, 2 to 3 scenarios per cell).

Simulator prediction for batch 2, computed before any batch-2 emulation run (`experiments/sim_replay_tcp_scenarios.py runs/emu_tcp_b b01 ... b18`,
output `experiments/final/sim_replay_emu_tcp_b.json`): learned - tuned max-utilisation -0.0087 [-0.0156, -0.0030], lower in 12 of 18, Wilcoxon p = 0.021;
learned - ECMP -0.0365; tuned - ECMP -0.0277.
