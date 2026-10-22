"""DOTE with the architecture and loss of the authors' released code (github.com/PredWanTE/DOTE, dote.py).

Kept from the release: history length 12, a five-layer MLP (four hidden layers of 128 ReLU units) over the flattened
demand history, a sigmoid output with one weight per candidate path, per-commodity normalisation of the weights into
split ratios, the exact maximum link utilisation of the true demand as the loss (each sample's loss is divided by
its own detached value, as in the release) and Adam at its default learning rate.

Differences from the release, all forced by the evaluation setting: the candidate paths are the repository's k = 4
shortest paths (the release uses 8 tunnels), the input is the controller's impaired demand estimate, demands are
divided by the mean link capacity, and mini-batches of 64 average the per-sample losses.
"""
import numpy as np
import torch
import torch.nn as nn

from stale_but_safe.baselines.heuristics import Policy

HIST = 12


class DoteNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, out_dim), nn.Sigmoid())

    def forward(self, x):
        return self.net(x.flatten(1))


def splits_from_weights(w, mask, D, k):
    """(B, D*k) sigmoid weights -> (B, D, k) split ratios over each demand's existing paths."""
    w = w.view(-1, D, k) * mask[None] + 1e-16
    return w * mask[None] / (w * mask[None]).sum(-1, keepdim=True)


def max_util(splits, dem, inc, cap):
    x = (dem[:, :, None] * splits).reshape(len(dem), -1)
    return (x @ inc.t() / cap).max(-1)[0]


def dote_loss(splits, dem, inc, cap):
    """Mean over the batch of max_cong / max_cong.detach() (the release's per-sample normalisation)."""
    mu = max_util(splits, dem, inc, cap)
    return (mu / mu.detach().clamp(min=1e-12)).mean(), mu.mean().item()


class DoteOfficialPolicy(Policy):
    def __init__(self, model, topo, name="dote-official"):
        self.m, self.name, self.unit = model, name, float(topo.cap.mean())
        self.mask = torch.from_numpy(topo.path_mask.astype(np.float32))

    def reset(self, env):
        super().reset(env)
        self.hist = []

    def act(self, obs, env):
        self.hist.append(np.asarray(obs["demand"], np.float32) / self.unit)
        h = self.hist[-HIST:]
        h = [h[0]] * (HIST - len(h)) + h
        with torch.no_grad():
            w = self.m(torch.from_numpy(np.concatenate(h))[None])
            sp = splits_from_weights(w, self.mask, self.topo.D, self.topo.k)[0].numpy().astype(float)
        sp = np.where(self.topo.valid_paths(obs["link_up"]), sp, 0.0)
        return sp / np.maximum(sp.sum(1, keepdims=True), 1e-9)
