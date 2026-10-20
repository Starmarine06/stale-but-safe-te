"""Extra TCP-emulation scenarios on new topology instances: python experiments/make_tcp_scenarios.py"""
from pathlib import Path
from stale_but_safe.emu import scenario as S
from stale_but_safe.env.te_env import EnvCfg
from stale_but_safe.env.topology import abilene, synthetic

SPECS = [("n14", "ba", 16, 2, 14), ("n15", "geo", 14, 2, 15), ("n16", "ba", 16, 3, 16),
         ("n17", "geo", 14, 3, 17), ("n18", "abilene", 0, 0, 18), ("n19", "ba", 16, 4, 19)]
for i, (name, kind, n, tseed, seed) in enumerate(SPECS):
    topo = abilene() if kind == "abilene" else synthetic(kind, n, seed=tseed)
    sc = S.make_scenario(topo, EnvCfg(load=0.8, horizon=90), seed, 90, active_n=60, delta=1.0, mbps_per_unit=4.0)
    out = Path("runs/emu_tcp") / name
    out.mkdir(parents=True, exist_ok=True)
    S.save(sc, out / "scenario.json")
    (out / "model.txt").write_text(f"runs/ours_dfm_final_s{i % 3 + 1}.pt")
    print(name, topo.name, len(sc["active"]))
