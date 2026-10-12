"""Topology Zoo loader (graphml -> Topology) and the fixed train/val/test split."""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import yaml

from .topology import Topology, degree_capacities, from_graph

ROOT = Path(__file__).resolve().parents[3]
ZOO_DIR = ROOT / "data" / "topologies" / "zoo_raw"
SPLITS = ROOT / "configs" / "splits.yaml"


def load_graph(path) -> nx.Graph | None:
    try:
        g = nx.read_graphml(str(path))
    except Exception:
        return None
    h = nx.Graph()
    for u, v in g.edges():
        if u != v:
            h.add_edge(u, v)
    if h.number_of_nodes() == 0:
        return None
    h = h.subgraph(max(nx.connected_components(h), key=len)).copy()
    degree_capacities(h)
    for u, v in h.edges:
        h.edges[u, v]["prop"] = 2.0
    return h


def catalog(min_n=10, max_n=50) -> dict[str, tuple[int, int]]:
    out = {}
    for f in sorted(ZOO_DIR.rglob("*.graphml")):
        g = load_graph(f)
        if g is not None and min_n <= g.number_of_nodes() <= max_n and g.number_of_edges() >= g.number_of_nodes():
            out[f.stem] = (g.number_of_nodes(), g.number_of_edges())
    return out


def make_splits(seed: int = 12345, n_val: int = 12) -> dict:
    """Validation = small/medium graphs (dev tuning). Test = held-out, incl. larger-than-train."""
    cat = catalog()
    names = sorted(cat)
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    small = [n for n in names if cat[n][0] <= 25]
    val = sorted(small[:n_val])
    test = sorted(n for n in names if n not in val)
    by_size = sorted(test, key=lambda n: cat[n][0])
    test_subset = sorted(by_size[i] for i in np.linspace(0, len(by_size) - 1, 24).astype(int))
    return {"seed": seed, "val": val, "test": test, "test_subset": test_subset,
            "sizes": {n: list(cat[n]) for n in val + test}}


def write_splits():
    sp = make_splits()
    SPLITS.parent.mkdir(exist_ok=True)
    SPLITS.write_text(yaml.safe_dump(sp, sort_keys=False))
    return sp


def load_split(split: str) -> list[str]:
    return yaml.safe_load(SPLITS.read_text())[split]


def zoo_topology(name: str, k: int = 4) -> Topology:
    f = next(ZOO_DIR.rglob(f"{name}.graphml"))
    return from_graph(load_graph(f), k=k, name=name)
