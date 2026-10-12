"""Turn the impaired observation into graph tensors (topology-agnostic)."""
from __future__ import annotations

from collections import deque

import numpy as np
import torch

from ..env.topology import Topology
from ..env.traffic import ecmp_splits


class GraphStatic:
    """Per-topology constants shared by all samples."""

    def __init__(self, topo: Topology):
        self.topo = topo
        E, ends = topo.E, np.array(topo.links)
        share = (ends[:, None, :, None] == ends[None, :, None, :]).any((2, 3))
        np.fill_diagonal(share, False)
        deg = np.maximum(share.sum(1, keepdims=True), 1)
        self.A = torch.tensor(share / deg, dtype=torch.float32)             # (E,E) mean agg
        hops = np.maximum(topo.inc.sum(0), 1)
        self.inc = torch.tensor(topo.inc, dtype=torch.float32)               # (E,Dk)
        self.inc_mean = torch.tensor(topo.inc / hops, dtype=torch.float32)
        mincap = np.where(topo.inc > 0, topo.cap[:, None], np.inf).min(0)
        mincap[~np.isfinite(mincap)] = topo.cap.max()
        self.mincap = mincap
        self.capref = float(np.median(topo.cap))
        self.logcap = torch.tensor(np.log(topo.cap / self.capref), dtype=torch.float32)
        self.hops = hops
        self.static_path = np.stack([hops / 6.0, np.log(mincap / self.capref)], 1).astype(np.float32)
        self.inc_np = topo.inc


def dims(H: int, use_age: bool) -> tuple[int, int]:
    """(n_link_features, n_path_features)"""
    return H + (H if use_age else 0) + 4, (7 if use_age else 6) + 2


class Featurizer:
    """Stateful: keeps the last H observations (the history the agent sees) and the
    controller's own last installed splits (known to it, so not telemetry)."""

    def __init__(self, topo: Topology, H: int = 3, use_age: bool = True, G: GraphStatic | None = None,
                 use_model: bool = True):
        self.G = G or GraphStatic(topo)
        self.topo, self.H, self.use_age, self.use_model = topo, H, use_age, use_model
        self.hist: deque = deque(maxlen=H)
        self.n_link, self.n_path = dims(H, use_age)
        self.prev = ecmp_splits(topo)

    def reset(self):
        self.hist.clear()
        self.prev = ecmp_splits(self.topo)

    def note_action(self, splits: np.ndarray):
        self.prev = np.asarray(splits, float)

    def _whatif(self, base_util, dem, x_prev):
        """Per path: max over its links of (util without this demand's own traffic
        + this demand's full load), i.e. the bottleneck if the demand moved there."""
        G, topo = self.G, self.topo
        E, k, D = topo.E, topo.k, topo.D
        own = (G.inc_np * (dem[:, None] * x_prev).ravel()[None, :]).reshape(E, D, k).sum(-1)  # (E,D) load
        capc = topo.cap[:, None]
        base = base_util[:, None] - own / capc                                                  # (E,D)
        base = np.repeat(base, k, axis=1)                                                       # (E,Dk)
        add = np.repeat(dem, k)[None, :] / capc
        w = (base + add) * G.inc_np
        return w.max(0)

    def __call__(self, obs: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        G, topo = self.G, self.topo
        up = obs["link_up"]
        u = np.clip(obs["util"], 0, 5).astype(np.float32)
        a = np.clip(obs["util_age"], 0, 20).astype(np.float32) / 10.0
        self.hist.append((u, a))
        while len(self.hist) < self.H:
            self.hist.appendleft(self.hist[0])
        dem = np.clip(obs["demand"], 0, None)
        valid = topo.valid_paths(up)
        ecmp = ecmp_splits(topo, valid)
        prev = np.where(valid, self.prev, 0.0)
        prev = prev / np.maximum(prev.sum(1, keepdims=True), 1e-9)
        pred_e = np.clip(G.inc_np @ (dem[:, None] * ecmp).ravel() / topo.cap, 0, 5).astype(np.float32)
        pred_p = np.clip(G.inc_np @ (dem[:, None] * prev).ravel() / topo.cap, 0, 5).astype(np.float32)
        if not self.use_model:   # ablation: hide all fluid-model-derived features
            pred_e = np.zeros_like(pred_e)
            pred_p = np.zeros_like(pred_p)
        us = [h[0] for h in self.hist]
        cols = us + ([h[1] for h in self.hist] if self.use_age else [])
        upf = up.astype(np.float32)
        xl = np.stack(cols + [pred_e, pred_p, upf, G.logcap.numpy()], 1)
        ui = (u * upf)[:, None] * G.inc_np
        pmax = ui.max(0)
        pmean = ui.sum(0) / G.hops
        dn = np.repeat(np.log1p(dem / G.capref), topo.k)
        pf = [pmax, pmean, dn, G.static_path[:, 0], G.static_path[:, 1], valid.ravel().astype(np.float32)]
        if self.use_age:
            pf.insert(3, np.repeat(np.clip(obs["demand_age"], 0, 20) / 10.0, topo.k))
        if self.use_model:
            wi_m = self._whatif(u * upf, dem, prev)
            wi_p = self._whatif(pred_p * upf, dem, prev)
        else:
            wi_m = wi_p = np.zeros(G.inc_np.shape[1])
        pf += [np.clip(wi_m, 0, 5), np.clip(wi_p, 0, 5)]
        xp = np.stack(pf, 1).astype(np.float32)
        return xl.astype(np.float32), xp, valid
