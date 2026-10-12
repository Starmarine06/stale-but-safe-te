"""Emulation scenario: a JSON-serialisable description shared by the simulator,
the Mininet runner and the Ryu controller, so all three see identical topology,
candidate paths, port numbering and demand schedule.

Units: capacities/demands are simulator units; `mbps_per_unit` converts to Mbit/s.
Interval `i` of the schedule lasts `delta` seconds.
"""
from __future__ import annotations

import json

import numpy as np

from ..env.te_env import EnvCfg, TEEnv
from ..env.topology import Topology
from ..env.traffic import ecmp_splits

WIRE_FACTOR = 1.046   # (payload 1000 B + UDP/IP/Eth headers + VLAN tag) / payload


def topo_to_dict(t: Topology) -> dict:
    return {"name": t.name, "n": t.n, "links": [list(l) for l in t.links], "cap": t.cap.tolist(),
            "prop": t.prop.tolist(), "demands": [list(d) for d in t.demands], "k": t.k,
            "path_links": t.path_links}


def topo_from_dict(d: dict) -> Topology:
    links = [tuple(l) for l in d["links"]]
    demands = [tuple(x) for x in d["demands"]]
    k, D, E = d["k"], len(demands), len(links)
    mask = np.zeros((D, k), bool)
    hops = np.full((D, k), 10 ** 6)
    inc = np.zeros((E, D * k))
    for di, paths in enumerate(d["path_links"]):
        for pi, ls in enumerate(paths):
            mask[di, pi] = True
            hops[di, pi] = len(ls)
            inc[ls, di * k + pi] = 1.0
    return Topology(d["name"], d["n"], links, np.array(d["cap"]), np.array(d["prop"]),
                    demands, k, d["path_links"], mask, hops, inc)


def host_ip(i: int) -> str:
    return f"10.0.0.{i + 1}"


def host_mac(i: int) -> str:
    return "00:00:00:00:00:%02x" % (i + 1)


def port_map(topo: Topology) -> dict:
    """Physical ports: port 1 = attached host; undirected link m (links 2m, 2m+1) gets
    the next free port on each end. Returns {'out_port': {link_id: port at tail switch},
    'link_ports': [(a, port_a, b, port_b)]} """
    nxt = {i: 2 for i in range(topo.n)}
    out_port, phys = {}, []
    for m in range(topo.E // 2):
        a, b = topo.links[2 * m]
        pa, pb = nxt[a], nxt[b]
        nxt[a] += 1
        nxt[b] += 1
        out_port[2 * m], out_port[2 * m + 1] = pa, pb
        phys.append((a, pa, b, pb))
    return {"out_port": out_port, "link_ports": phys}


def make_scenario(topo: Topology, cfg: EnvCfg, seed: int, steps: int, active_n: int = 30,
                  delta: float = 1.0, mbps_per_unit: float = 4.0) -> dict:
    """Pick the `active_n` heaviest base demands, then record the exact demand
    schedule the simulator would generate (same seed => same traffic)."""
    probe = TEEnv(topo, EnvCfg(load=cfg.load, horizon=steps, traffic=cfg.traffic), seed)
    order = np.argsort(-probe.traffic.base)
    active = np.zeros(topo.D, bool)
    active[order[:active_n]] = True
    tcfg = {**cfg.traffic, "active": active}
    env_cfg = EnvCfg(load=cfg.load, horizon=steps, telemetry=cfg.telemetry, L=cfg.L,
                     traffic=tcfg, fail_at=cfg.fail_at)
    env = TEEnv(topo, env_cfg, seed)
    env.reset(seed)
    sched = []
    for _ in range(steps):
        sched.append(env.peek_demand().copy())
        env.step(ecmp_splits(topo))
    sched = np.array(sched)
    pm = port_map(topo)
    return {"topo": topo_to_dict(topo), "seed": seed, "steps": steps, "delta": delta,
            "mbps_per_unit": mbps_per_unit, "active": np.flatnonzero(active).tolist(),
            "schedule": sched.tolist(), "ports": {"out_port": {str(k): v for k, v in pm["out_port"].items()},
                                                  "link_ports": pm["link_ports"]},
            "load": cfg.load, "traffic": {k: v for k, v in cfg.traffic.items()}}


def sim_cfg(sc: dict, telemetry, L: int = 0, carried: bool = False) -> EnvCfg:
    """Simulator config that reproduces the scenario's traffic exactly."""
    active = np.zeros(len(sc["topo"]["demands"]), bool)
    active[sc["active"]] = True
    return EnvCfg(load=sc["load"], horizon=sc["steps"], telemetry=telemetry, L=L, carried_obs=carried,
                  traffic={**sc["traffic"], "active": active})


def save(sc: dict, path):
    with open(path, "w") as f:
        json.dump(sc, f)


def load(path) -> dict:
    with open(path) as f:
        return json.load(f)
