"""The max-utilisation loss and weight normalisation match the authors' released code (skipped if it is not cloned)."""
import ast
import pathlib

import numpy as np
import pytest
import torch

from stale_but_safe.baselines.dote_official import max_util, splits_from_weights
from stale_but_safe.env.topology import synthetic

SRC = pathlib.Path(__file__).resolve().parents[1] / "external" / "DOTE" / "dote.py"


def _release_loss():
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "loss_fn_maxutil")
    ns = {"torch": torch, "np": np}
    exec(compile(ast.Module([fn], []), str(SRC), "exec"), ns)
    return ns["loss_fn_maxutil"], ns


@pytest.mark.skipif(not SRC.exists(), reason="external/DOTE not cloned")
def test_loss_matches_release():
    topo = synthetic("ba", 8, 1)
    D, k, E = topo.D, topo.k, topo.E
    cols = [d * k + p for d in range(D) for p in range(k) if topo.path_mask[d, p]]
    P = len(cols)
    c2p = np.zeros((D, P))
    for j, c in enumerate(cols):
        c2p[c // k, j] = 1.0
    p2e = topo.inc[:, cols].T.astype(float)
    fn, ns = _release_loss()
    ns["commodities_to_paths"] = torch.tensor(c2p)
    ns["paths_to_edges"] = torch.tensor(p2e)

    class Env:
        _capacities = topo.cap.astype(float)

        def get_num_nodes(self):
            return topo.n

    # release assumes num_nodes*(num_nodes-1) commodities; our synthetic graph is all-pairs
    assert D == topo.n * (topo.n - 1)
    rng = np.random.default_rng(0)
    dem = rng.uniform(0.1, 1.0, (1, D))
    w_rel = rng.uniform(0.05, 1.0, (1, P))
    y_true = torch.tensor(np.concatenate([dem, [[1.0]]], axis=1))     # opt = 1 -> loss_val is max congestion
    _, val_release = fn(torch.tensor(w_rel), y_true, Env())

    w_full = np.zeros((1, D * k))
    w_full[0, cols] = w_rel[0]
    mask = torch.from_numpy(topo.path_mask.astype(np.float64))
    sp = splits_from_weights(torch.tensor(w_full), mask, D, k)
    mine = max_util(sp, torch.tensor(dem), torch.tensor(topo.inc, dtype=torch.float64), torch.tensor(topo.cap, dtype=torch.float64))
    assert mine.item() == pytest.approx(val_release, rel=1e-9)
