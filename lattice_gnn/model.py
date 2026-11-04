"""Graph neural network: periodic lattice graph -> 9 orthotropic stiffness targets.

Architecture (message passing on nodes *and* struts):
  1. Encode node features (coordination) and strut features (length, direction).
  2. Repeat `layers` times: update each strut from its two end nodes, then update
     each node from the sum of its struts. Residual connections + LayerNorm.
  3. Pool over the cell with mean and max (not sum, so the result doesn't depend
     on how big a unit cell the file happens to use), add the global features,
     and map to the 9 targets with an MLP.
  4. Optional `pd_head`: the last 3 outputs are not the couplings themselves but
     unconstrained numbers mapped to couplings that always give a positive definite
     stiffness (see `pd_couplings`).

All inputs and targets are standardized with statistics from the training set;
those statistics are stored in the model so a saved checkpoint is self-contained.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.utils import scatter


# Positive-definite coupling head.
# The stiffness is positive definite exactly when the 3x3 normalized compliance block
# R = [[1, -c_xy, -c_xz], [-c_xy, 1, -c_yz], [-c_xz, -c_yz, 1]] is (lattice_gnn.evaluate explains
# why). R has a unit diagonal, so it is a correlation matrix, and every positive semidefinite
# correlation matrix is R = L L^T with L lower triangular and unit-length rows. For 3x3 that
# L is built from three numbers in [-1, 1] (canonical partial correlations, Lewandowski et al.
# 2009, "Generating random correlation matrices based on vines"):
#     L = [[1,   0,                      0                              ],
#          [r01, s01,                    0                              ],
#          [r02, p12 * s02,              sqrt(1 - r02^2) sqrt(1 - p12^2)]],   s = sqrt(1 - r^2)
# giving R01 = r01, R02 = r02, R12 = r01 r02 + s01 s02 p12. Each number is tanh(a) of a network
# output a, so any a gives a valid R. tanh reaches exactly 1 in float32 (|a| > ~9), where R is
# only semidefinite, so the head returns (1 - PD_MARGIN) R + PD_MARGIN I: its smallest eigenvalue
# is at least PD_MARGIN for every input in exact arithmetic. Computing R12 in float32 rounds each
# coupling by ~eps32, which (Weyl's inequality, as in evaluate.PD_TOL) moves the eigenvalue by
# well under 1e-6; tests/test_pd_head.py measures ~8e-8.
# PD_MARGIN: 1e-5 is ~28x the evaluator's tolerance (evaluate.PD_TOL = 3 * eps32 = 3.6e-7), so
# that rounding cannot push a prediction below it, and it stays below the true labels: on
# the full catalog only 4 of 17,222 labels have a smallest eigenvalue under 1e-4 (median 0.037,
# 1st percentile 0.002), and those 4 are the (near-)non-PD labels the model cannot match anyway.
PD_MARGIN = 1e-5
PD_CLAMP = 15.0   # tanh(15) == 1 in float32; clamping keeps 1/cosh and its gradient finite


def pd_couplings(a: torch.Tensor) -> torch.Tensor:
    """(n, 3) unconstrained outputs -> (n, 3) couplings [c_yz, c_xz, c_xy] (graph.TARGET_NAMES order)
    whose normalized compliance block is positive definite, with smallest eigenvalue >= PD_MARGIN
    (less float32 rounding, < 1e-6).

    Axes x, y, z are 0, 1, 2. a[:, 2] sets the x-y correlation, a[:, 1] the x-z correlation and
    a[:, 0] the y-z partial correlation given x.
    """
    a = a.clamp(-PD_CLAMP, PD_CLAMP)
    r = torch.tanh(a)
    s = 1 / torch.cosh(a)              # == sqrt(1 - tanh^2), without the sqrt's infinite gradient at 1
    r01, r02, p12 = r[:, 2], r[:, 1], r[:, 0]
    s01, s02 = s[:, 2], s[:, 1]
    r12 = r01 * r02 + s01 * s02 * p12
    return -(1 - PD_MARGIN) * torch.stack([r12, r02, r01], dim=1)


def mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
    mods = []
    for i in range(len(dims) - 1):
        mods.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            mods.append(nn.SiLU())
    return nn.Sequential(*mods)


class Standardizer(nn.Module):
    """(x - mean) / std, with mean and std saved alongside the model weights."""

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    def fit(self, values: torch.Tensor):
        std = values.std(0) if len(values) > 1 else torch.ones_like(self.std)
        self.mean.copy_(values.mean(0))
        self.std.copy_(torch.where(std > 1e-6, std, torch.ones_like(std)))   # constant features

    def forward(self, values):
        return (values - self.mean) / self.std

    def inverse(self, values):
        return values * self.std + self.mean


class MessagePassingLayer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.edge_mlp = mlp(3 * hidden, hidden, hidden)
        self.node_mlp = mlp(2 * hidden, hidden, hidden)
        self.edge_norm = nn.LayerNorm(hidden)
        self.node_norm = nn.LayerNorm(hidden)

    def forward(self, h, e, edge_index):
        src, dst = edge_index
        e = self.edge_norm(e + self.edge_mlp(torch.cat([h[src], h[dst], e], dim=1)))
        incoming = scatter(e, dst, dim=0, dim_size=h.size(0), reduce="sum")
        h = self.node_norm(h + self.node_mlp(torch.cat([h, incoming], dim=1)))
        return h, e


class LatticeGNN(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, global_dim: int, out_dim: int,
                 hidden: int = 128, layers: int = 4, pd_head: bool = False):
        super().__init__()
        if pd_head and out_dim != 9:
            raise ValueError("pd_head needs the 9 orthotropic targets (6 log10 moduli, 3 couplings)")
        self.config = dict(node_dim=node_dim, edge_dim=edge_dim, global_dim=global_dim,
                           out_dim=out_dim, hidden=hidden, layers=layers, pd_head=pd_head)
        self.pd_head = pd_head
        self.x_norm = Standardizer(node_dim)
        self.e_norm = Standardizer(edge_dim)
        self.u_norm = Standardizer(global_dim)
        self.y_norm = Standardizer(out_dim)
        self.node_enc = mlp(node_dim, hidden, hidden)
        self.edge_enc = mlp(edge_dim, hidden, hidden)
        self.layers = nn.ModuleList(MessagePassingLayer(hidden) for _ in range(layers))
        self.head = mlp(4 * hidden + global_dim, hidden, out_dim, layers=3)

    def fit_normalizers(self, graphs: list):
        self.x_norm.fit(torch.cat([g.x for g in graphs]))
        self.e_norm.fit(torch.cat([g.edge_attr for g in graphs]))
        self.u_norm.fit(torch.cat([g.u for g in graphs]))
        self.y_norm.fit(torch.cat([g.y for g in graphs]))

    def _head(self, data):
        """Raw head output: standardized targets, or with pd_head 6 standardized log10 moduli
        and 3 unconstrained coupling parameters."""
        h = self.node_enc(self.x_norm(data.x))
        e = self.edge_enc(self.e_norm(data.edge_attr))
        for layer in self.layers:
            h, e = layer(h, e, data.edge_index)

        batch = data.batch if data.batch is not None else h.new_zeros(h.size(0), dtype=torch.long)
        edge_batch = batch[data.edge_index[0]]
        size = data.u.size(0)
        pooled = torch.cat([
            global_mean_pool(h, batch, size), global_max_pool(h, batch, size),
            global_mean_pool(e, edge_batch, size), global_max_pool(e, edge_batch, size),
            self.u_norm(data.u),
        ], dim=1)
        return self.head(pooled)

    def forward(self, data):
        """Standardized predictions; the training loss is computed on these."""
        out = self._head(data)
        if not self.pd_head:
            return out
        couplings = pd_couplings(out[:, 6:])
        return torch.cat([out[:, :6], (couplings - self.y_norm.mean[6:]) / self.y_norm.std[6:]], dim=1)

    @torch.no_grad()
    def predict(self, data):
        """Predictions in target units (see lattice_gnn.graph.TARGET_NAMES)."""
        out = self._head(data)
        if not self.pd_head:
            return self.y_norm.inverse(out)
        # Couplings straight from pd_couplings, not through the standardizer, so float32
        # rounding in (c - mean) / std * std + mean cannot eat into PD_MARGIN.
        moduli = out[:, :6] * self.y_norm.std[:6] + self.y_norm.mean[:6]
        return torch.cat([moduli, pd_couplings(out[:, 6:])], dim=1)

    def save(self, path: str | Path, **extra):
        torch.save({"config": self.config, "state_dict": self.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path: str | Path) -> tuple[LatticeGNN, dict]:
        """Returns the model (in eval mode) and the full checkpoint dict."""
        checkpoint = torch.load(path, weights_only=False)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model, checkpoint
