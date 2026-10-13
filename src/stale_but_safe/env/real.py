"""Real networks and traffic from SNDlib (Abilene/Zhang, GEANT/Uhlig, Germany-DFN).

`python -m stale_but_safe.env.real prepare` converts the downloaded archives in `data/real/` into one
compact `data/real/<name>.npz` per network: capacities (Mbit/s), node order and a (T, D) demand trace
(Mbit/s) in the Topology's own demand order. `load(name)` returns (Topology, trace).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import networkx as nx
import numpy as np

from .topology import Topology, from_graph

ROOT = Path(__file__).resolve().parents[3] / "data" / "real"
NETS = {   # name -> (network file, trace directory glob)
    "abilene": ("abilene", "abilene"),
    "geant": ("geant", "geant"),
    "germany17": ("nobel-germany", "nobel"),
    "germany50": ("germany50", "germany50"),
    "brain": ("brain", "brain"),      # 161 nodes: only the heaviest TOP_DEMANDS pairs are kept
}
TOP_DEMANDS = {"brain": 600}
_LINK = re.compile(r"^\s*(\S+)\s*\(\s*(\S+)\s+(\S+)\s*\)\s+([\d.]+)")
_DEM = re.compile(r"^\s*\S+\s*\(\s*(\S+)\s+(\S+)\s*\)\s+\d+\s+([\d.eE+-]+)")


def _section(text: str, name: str) -> list[str]:
    lines = text.splitlines()
    out, on = [], False
    for ln in lines:
        if not on and ln.strip().startswith(name + " ("):
            on = True
            continue
        if on:
            if ln.strip() == ")":
                break
            out.append(ln)
    return out


def parse_network(path: Path):
    """-> (nodes in file order, [(u, v, capacity_mbps)])."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    nodes = [ln.split()[0] for ln in _section(text, "NODES") if ln.strip() and not ln.strip().startswith("#")]
    links = []
    for ln in _section(text, "LINKS"):
        m = _LINK.match(ln)
        if m:
            # first number after the endpoints is the pre-installed capacity; fall back to module capacity
            cap = float(m.group(4))
            if cap <= 0:
                tail = ln[ln.rfind("(") + 1:] if ln.count("(") > 1 else ""
                nums = re.findall(r"[\d.]+", tail)
                cap = float(nums[0]) if nums else 1.0
            links.append((m.group(2), m.group(3), cap))
    return nodes, links


def parse_tm(path: Path) -> dict[tuple[str, str], float]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    out = {}
    for ln in _section(text, "DEMANDS"):
        m = _DEM.match(ln)
        if m:
            out[(m.group(1), m.group(2))] = float(m.group(3))
    return out


def build_topology(name: str, k: int = 4, demands=None) -> tuple[Topology, list[str]]:
    net, _ = NETS[name]
    if demands is None and (ROOT / f"{name}.npz").exists():
        z = np.load(ROOT / f"{name}.npz")
        if "demands" in z.files:
            demands = z["demands"].tolist()
    nodes, links = parse_network(ROOT / "nets" / f"{net}.txt")
    g = nx.Graph()
    g.add_nodes_from(nodes)
    for u, v, c in links:
        g.add_edge(u, v, cap=c, prop=1.0)
    topo = from_graph(g, k=k, name=name, demands=demands)
    return topo, nodes


def _matrices(archive, nodes, every: int = 1):
    """Yield (file name, {(src_idx, dst_idx): value}) straight from the .tgz."""
    import tarfile
    idx = {n: i for i, n in enumerate(nodes)}
    with tarfile.open(archive, "r:gz") as tf:
        for k, m in enumerate(tf):
            if not (m.isfile() and "demandMatrix" in m.name) or (k % every):
                continue
            text = tf.extractfile(m).read().decode("utf-8", "ignore")
            out = {}
            for ln in _section(text, "DEMANDS"):
                mm = _DEM.match(ln)
                if mm and mm.group(1) in idx and mm.group(2) in idx and mm.group(1) != mm.group(2):
                    out[(idx[mm.group(1)], idx[mm.group(2)])] = float(mm.group(3))
            yield m.name, out


def prepare(name: str, limit: int | None = None) -> Path:
    """Stream the demand matrices straight out of the downloaded .tgz (no extraction needed).
    Big networks keep only the TOP_DEMANDS heaviest pairs (by mean volume, chosen on a 1-in-20 sample)."""
    _, tdir = NETS[name]
    archive = next(ROOT.glob(f"directed-{tdir}*-native.tgz"))
    _, nodes = build_topology(name, demands=[])      # node order only
    demands = None
    if name in TOP_DEMANDS:
        acc = {}
        for i, (_, tm) in enumerate(_matrices(archive, nodes, every=20)):
            for k, v in tm.items():
                acc[k] = acc.get(k, 0.0) + v
        demands = [list(k) for k, _ in sorted(acc.items(), key=lambda kv: -kv[1])[:TOP_DEMANDS[name]]]
        share = sum(acc[tuple(d)] for d in demands) / sum(acc.values())
        print(f"{name}: kept {len(demands)} heaviest demands = {100 * share:.1f}% of sampled traffic", flush=True)
    topo, _ = build_topology(name, demands=demands)
    pos = {(s, t): j for j, (s, t) in enumerate(topo.demands)}
    rows, names = [], []
    for fname, tm in _matrices(archive, nodes):
        row = np.zeros(topo.D, np.float32)
        for k, v in tm.items():
            j = pos.get(k)
            if j is not None:
                row[j] = v
        rows.append(row)
        names.append(fname)
        if len(rows) % 2000 == 0:
            print(name, len(rows), flush=True)
        if limit and len(rows) >= limit:
            break
    order = np.argsort(names)                      # file names are timestamps
    trace = np.stack(rows)[order]
    out = ROOT / f"{name}.npz"
    extra = {"demands": np.array(topo.demands)} if demands is not None else {}
    np.savez_compressed(out, trace=trace, nodes=np.array(nodes), cap=topo.cap, **extra)
    return out


def load(name: str, k: int = 4):
    """Real topology plus its demand trace (T, D) in Mbit/s; capacities are in Mbit/s too."""
    topo, _ = build_topology(name, k)
    z = np.load(ROOT / f"{name}.npz")
    return topo, z["trace"].astype(float)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "prepare":
        for n in (sys.argv[2:] or list(NETS)):
            print(prepare(n))
