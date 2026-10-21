"""Batch-2 TCP scenarios (design: docs/tcp_batch2_design.md): python experiments/make_tcp_scenarios2.py -> runs/emu_tcp_b/<id>/"""
from pathlib import Path
from stale_but_safe.emu import scenario as S
from stale_but_safe.env.te_env import EnvCfg
from stale_but_safe.env.topology import abilene, synthetic

GRAV, HOT = {}, {"hotspot_prob": 0.1}
SHIFT, FLASH = {"shift_at": 45}, {"flash_at": 30, "flash_len": 20}
SPECS = ([("ba", 5 + i, GRAV, 0.8) for i in range(3)] + [("ba", 5 + i, HOT, 0.8) for i in range(3)] +
         [("geo", 5 + i, SHIFT, 0.8) for i in range(3)] + [("geo", 5 + i, FLASH, 0.8) for i in range(3)] +
         [("ba", 8, GRAV, 0.6), ("geo", 8, GRAV, 0.6), ("ba", 9, GRAV, 1.0), ("geo", 9, GRAV, 1.0),
          ("abilene", 0, HOT, 0.8), ("abilene", 0, FLASH, 0.8)])
for i, (kind, tseed, traffic, load) in enumerate(SPECS):
    name = f"b{i + 1:02d}"
    topo = abilene() if kind == "abilene" else synthetic(kind, 16 if kind == "ba" else 14, seed=tseed)
    sc = S.make_scenario(topo, EnvCfg(load=load, horizon=90, traffic=traffic), 100 + i, 90, active_n=60, delta=1.0, mbps_per_unit=4.0)
    out = Path("runs/emu_tcp_b") / name
    out.mkdir(parents=True, exist_ok=True)
    S.save(sc, out / "scenario.json")
    (out / "model.txt").write_text(f"runs/ours_dfm_final_s{i % 3 + 1}.pt")
    print(name, topo.name, kind, traffic, load, len(sc["active"]))
