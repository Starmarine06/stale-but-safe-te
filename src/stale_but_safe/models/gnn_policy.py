"""Topology-agnostic GNN policy: link message passing -> path pooling -> shared
path scorer -> softmax over each demand's k candidate paths."""
from __future__ import annotations

import torch
import torch.nn as nn


class GNNPolicy(nn.Module):
    def __init__(self, n_link: int, n_path: int, k: int = 4, hid: int = 64,
                 layers: int = 3, use_gnn: bool = True):
        super().__init__()
        self.k, self.use_gnn, self.layers = k, use_gnn, layers
        self.enc = nn.Sequential(nn.Linear(n_link, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU())
        self.self_l = nn.ModuleList(nn.Linear(hid, hid) for _ in range(layers))
        self.nb_l = nn.ModuleList(nn.Linear(hid, hid, bias=False) for _ in range(layers))
        self.norm = nn.ModuleList(nn.LayerNorm(hid) for _ in range(layers))
        self.score = nn.Sequential(nn.Linear(2 * hid + n_path, hid), nn.ReLU(),
                                   nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))
        self.value = nn.Sequential(nn.Linear(hid + 2, hid), nn.ReLU(), nn.Linear(hid, 1))
        self.log_std = nn.Parameter(torch.full((1,), -1.0))

    def embed(self, xl, G):
        h = self.enc(xl)
        if self.use_gnn:
            for sl, nl, nm in zip(self.self_l, self.nb_l, self.norm):
                h = nm(h + torch.relu(sl(h) + nl(G.A @ h)))
        return h

    def forward(self, xl, xp, valid, G):
        """xl (B,E,Fl) xp (B,Dk,Fp) valid (B,D,k) bool -> logits (B,D,k), value (B,)"""
        B, Dk, _ = xp.shape
        D = Dk // self.k
        h = self.embed(xl, G)                                               # (B,E,H)
        pe = torch.matmul(G.inc_mean.t(), h)                                # (B,Dk,H)
        pe4 = pe.view(B, D, self.k, -1)
        vm = valid.unsqueeze(-1).float()
        ctx = (pe4 * vm).sum(2, keepdim=True) / vm.sum(2, keepdim=True).clamp(min=1)
        z = torch.cat([pe, ctx.expand_as(pe4).reshape(B, Dk, -1), xp], -1)
        logits = self.score(z).view(B, D, self.k)
        logits = logits.masked_fill(~valid, -1e9)
        g = torch.cat([h.mean(1), xl[:, :, 0].max(1, keepdim=True)[0], xl[:, :, 0].mean(1, keepdim=True)], -1)
        return logits, self.value(g).squeeze(-1)
