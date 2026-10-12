"""Wrap a trained GNNPolicy as a baselines.Policy; build/save/load models."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from ..baselines.heuristics import Policy
from .features import Featurizer, GraphStatic, dims
from .gnn_policy import GNNPolicy


@dataclass
class ModelCfg:
    H: int = 3
    use_age: bool = True
    use_gnn: bool = True
    use_model: bool = True
    hid: int = 64
    layers: int = 3
    k: int = 4


def build(cfg: ModelCfg) -> GNNPolicy:
    n_link, n_path = dims(cfg.H, cfg.use_age)
    return GNNPolicy(n_link, n_path, cfg.k, cfg.hid, cfg.layers, cfg.use_gnn)


def save(model, cfg: ModelCfg, path, extra=None):
    torch.save({"cfg": asdict(cfg), "state": model.state_dict(), "extra": extra or {}}, path)


def load(path) -> tuple[GNNPolicy, ModelCfg]:
    d = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelCfg(**d["cfg"])
    m = build(cfg)
    m.load_state_dict(d["state"])
    m.eval()
    return m, cfg


_G: dict = {}


def graph_static(topo) -> GraphStatic:
    key = (topo.name, topo.n, topo.E)
    if key not in _G:
        _G[key] = GraphStatic(topo)
    return _G[key]


class LearnedPolicy(Policy):
    def __init__(self, model: GNNPolicy, cfg: ModelCfg, name="gnn"):
        self.model, self.cfg, self.name = model, cfg, name
        self.model.eval()

    def reset(self, env):
        super().reset(env)
        self.feat = Featurizer(env.topo, self.cfg.H, self.cfg.use_age, graph_static(env.topo), self.cfg.use_model)

    @torch.no_grad()
    def act(self, obs, env):
        xl, xp, valid = self.feat(obs)
        t = lambda a: torch.from_numpy(a)[None]
        logits, _ = self.model(t(xl), t(xp), t(valid), self.feat.G)
        out = torch.softmax(logits[0], -1).numpy()
        self.feat.note_action(out)
        return out
