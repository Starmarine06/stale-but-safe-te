"""Topology with k candidate paths per demand and a link/path incidence matrix."""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass
class Topology:
    name: str
    n: int
    links: list[tuple[int, int]]   # directed links, index = link id
    cap: np.ndarray                # (E,) capacity
    prop: np.ndarray               # (E,) propagation delay
    demands: list[tuple[int, int]]  # (s, t) pairs, index = demand id
    k: int
    path_links: list[list[list[int]]]  # [demand][path] -> link ids
    path_mask: np.ndarray          # (D, k) True where path exists
    hops: np.ndarray               # (D, k) hop count (large where missing)
    inc: np.ndarray                # (E, D*k) 1 if path uses link; column = d*k + p

    @property
    def E(self) -> int:
        return len(self.links)

    @property
    def D(self) -> int:
        return len(self.demands)

    def valid_paths(self, link_up: np.ndarray | None = None) -> np.ndarray:
        """(D, k) mask of paths that exist and use only up links."""
        if link_up is None:
            return self.path_mask.copy()
        down = (~link_up).astype(float)
        broken = (down @ self.inc).reshape(self.D, self.k) > 0
        return self.path_mask & ~broken


def from_graph(g: nx.Graph, k: int = 4, name: str = "graph",
               default_cap: float = 1.0) -> Topology:
    """Build from an undirected graph with optional edge attrs 'cap' and 'prop'."""
    g = nx.convert_node_labels_to_integers(g)
    n = g.number_of_nodes()
    links, cap, prop = [], [], []
    lid = {}
    for u, v, a in g.edges(data=True):
        for s, t in ((u, v), (v, u)):
            lid[(s, t)] = len(links)
            links.append((s, t))
            cap.append(a.get("cap", default_cap))
            prop.append(a.get("prop", 1.0))
    demands = [(s, t) for s in range(n) for t in range(n) if s != t]
    D = len(demands)
    path_links = []
    mask = np.zeros((D, k), bool)
    hops = np.full((D, k), 10**6)
    inc = np.zeros((len(links), D * k))
    for di, (s, t) in enumerate(demands):
        paths = []
        for pi, p in enumerate(nx.shortest_simple_paths(g, s, t)):
            if pi >= k:
                break
            ls = [lid[(a, b)] for a, b in zip(p[:-1], p[1:])]
            paths.append(ls)
            mask[di, pi] = True
            hops[di, pi] = len(ls)
            inc[ls, di * k + pi] = 1.0
        path_links.append(paths)
    return Topology(name, n, links, np.array(cap, float), np.array(prop, float),
                    demands, k, path_links, mask, hops, inc)


# ---------------------------------------------------------------- generators
def synthetic(kind: str, n: int, seed: int, k: int = 4,
              caps=(1.0, 2.5, 10.0)) -> Topology:
    """Random connected topology with heterogeneous capacities (units: Gbps)."""
    rng = np.random.default_rng(seed)
    for attempt in range(100):
        s = int(rng.integers(1 << 30))
        if kind == "ba":
            g = nx.barabasi_albert_graph(n, 2, seed=s)
        elif kind == "er":
            g = nx.erdos_renyi_graph(n, min(1.0, 3.0 / n + 0.1), seed=s)
        elif kind == "geo":
            g = nx.random_geometric_graph(n, 0.4, seed=s)
        else:
            raise ValueError(kind)
        if nx.is_connected(g):
            break
    else:
        raise RuntimeError("could not generate a connected graph")
    for u, v in g.edges:
        g.edges[u, v]["cap"] = float(rng.choice(caps))
        g.edges[u, v]["prop"] = float(rng.uniform(0.5, 5.0))
    return from_graph(g, k=k, name=f"{kind}{n}-s{seed}")


_ABILENE = [("ATLAM5", "ATLAng"), ("ATLAng", "HSTNng"), ("ATLAng", "IPLSng"),
            ("ATLAng", "WASHng"), ("CHINng", "IPLSng"), ("CHINng", "NYCMng"),
            ("DNVRng", "KSCYng"), ("DNVRng", "SNVAng"), ("DNVRng", "STTLng"),
            ("HSTNng", "KSCYng"), ("HSTNng", "LOSAng"), ("IPLSng", "KSCYng"),
            ("LOSAng", "SNVAng"), ("NYCMng", "WASHng"), ("SNVAng", "STTLng")]


def abilene(k: int = 4) -> Topology:
    g = nx.Graph()
    for u, v in _ABILENE:
        g.add_edge(u, v, cap=2.5 if "ATLAM5" in (u, v) else 10.0, prop=2.0)
    return from_graph(g, k=k, name="abilene")
