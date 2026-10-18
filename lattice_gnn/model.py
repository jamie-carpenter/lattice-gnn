"""Graph neural network: periodic lattice graph -> 9 orthotropic stiffness targets.

Architecture (message passing on nodes *and* struts):
  1. Encode node features (coordination) and strut features (length, direction).
  2. Repeat `layers` times: update each strut from its two end nodes, then update
     each node from the sum of its struts. Residual connections + LayerNorm.
  3. Pool over the cell with mean and max (not sum, so the result doesn't depend
     on how big a unit cell the file happens to use), add the global features,
     and map to the 9 targets with an MLP.

All inputs and targets are standardized with statistics from the training set;
those statistics are stored in the model so a saved checkpoint is self-contained.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.utils import scatter


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
                 hidden: int = 128, layers: int = 4):
        super().__init__()
        self.config = dict(node_dim=node_dim, edge_dim=edge_dim, global_dim=global_dim,
                           out_dim=out_dim, hidden=hidden, layers=layers)
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

    def forward(self, data):
        """Standardized predictions; the training loss is computed on these."""
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

    @torch.no_grad()
    def predict(self, data):
        """Predictions in target units (see lattice_gnn.graph.TARGET_NAMES)."""
        return self.y_norm.inverse(self(data))

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
