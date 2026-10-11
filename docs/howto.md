# How-to

## Run the tests
```
pip install -e ".[dev]"
python -m pytest -q
```

## Add a topology
* Topology Zoo graph: drop the `.graphml` file anywhere under `data/topologies/zoo_raw/` and use its
  file stem: `stale-but-safe demo --topology Geant2012` (capacities come from the degree rule in
  `env/topology.py::degree_capacities`; replace them if you have real capacities).
* Your own `networkx.Graph`: set optional edge attrs `cap` and `prop`, then
  `env.topology.from_graph(g, k=4, name="mynet")`.

## Plug in a policy
Subclass `baselines.heuristics.Policy`:
```python
class MyPolicy(Policy):
    name = "mine"
    def act(self, obs, env):      # obs keys: util, util_age, demand, demand_age, link_up, t
        return splits             # (D, k) non-negative; rows are re-normalised over valid paths
```
Use it in `eval.compare.table(...)`, wrap it in `guard.Guard(MyPolicy())`, or in the controller via a
new branch in `controller/ryu_app.py::build_policy`. Only `LPOracle` may use `env.peek_demand()`.

## Train
```
python -m stale_but_safe.train.bc  --mode random --out runs/ours.pt     # telemetry-randomised distillation
python -m stale_but_safe.train.bc  --mode ideal  --out runs/ideal.pt    # perfect-telemetry baseline
python -m stale_but_safe.train.ppo --init runs/ours.pt --out runs/ours_ppo.pt
```
Ablation flags for `bc`: `--H 1` (no history), `--no-age` (no age mask), `--no-gnn` (no message passing).
Each rollout worker imports torch (~1 GB of committed memory on Windows), so size `--workers` to your RAM.

## Run the emulation (needs Docker; Linux containers)
```
docker build -f docker/mininet.Dockerfile -t sbs-mininet .
docker build -f docker/ryu.Dockerfile     -t sbs-ryu .
stale-but-safe scenario --out runs/emu --topology emu10 --steps 60
python -m stale_but_safe.emu.run_emulation --scenario runs/emu/scenario.json \
    --run ecmp:snmp --run tuned:snmp:snmp --run model:runs/ours.pt:snmp --run guard0.05@model:runs/ours.pt:snmp
```
Each run starts a Ryu container and a privileged Mininet container (OVS kernel datapath, `tc` shaping)
and writes traces under `runs/emu/<tag>/`.

## Shadow mode demo
`stale-but-safe demo --topology abilene --mode shadow --model runs/ours.pt` – the policy proposes, ECMP runs,
and the tool reports what the policy would have predicted.
