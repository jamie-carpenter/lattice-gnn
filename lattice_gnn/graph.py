"""Turn a parsed Lattice into a periodic graph and GNN-ready tensors.

The unit cell in the file repeats nodes on its faces (e.g. all 8 corners of a
cube are the same node in the infinite lattice) and repeats bars on its faces.
build_periodic_graph merges these periodic copies so each node and bar of the
infinite lattice appears exactly once per cell. Bars that cross the cell
boundary keep an integer offset telling which neighboring cell they reach.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .parse import Lattice

MODULI = ("Ex", "Ey", "Ez", "Gyz", "Gxz", "Gxy")   # Voigt order: xx, yy, zz, yz, xz, xy
PAIRS = (("y", "z"), ("x", "z"), ("x", "y"))
TARGET_NAMES = ("log10_Ex", "log10_Ey", "log10_Ez", "log10_Gyz", "log10_Gxz", "log10_Gxy",
                "c_yz", "c_xz", "c_xy")
# What to call each target in printed results.
DISPLAY_NAMES = ("Ex", "Ey", "Ez", "Gyz", "Gxz", "Gxy", "Poisson yz", "Poisson xz", "Poisson xy")
MIN_MODULUS = 1e-12   # floor before log10, in case a modulus is reported as 0


def lattice_matrix(cell: np.ndarray) -> np.ndarray:
    """Rows are the lattice vectors a1, a2, a3 (a1 along x, a2 in the xy-plane)."""
    a, b, c = cell[:3]
    alpha, beta, gamma = np.radians(cell[3:])
    cx = c * np.cos(beta)
    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    cz = np.sqrt(c**2 - cx**2 - cy**2)
    return np.array([
        [a, 0.0, 0.0],
        [b * np.cos(gamma), b * np.sin(gamma), 0.0],
        [cx, cy, cz],
    ])


@dataclass
class PeriodicGraph:
    frac: np.ndarray      # (N, 3) unique node positions, fractional
    pos: np.ndarray       # (N, 3) cartesian positions
    edges: np.ndarray     # (E, 2) each bar of the infinite lattice once
    offsets: np.ndarray   # (E, 3) int cell shift: bar goes from frac[i] to frac[j] + offset
    vectors: np.ndarray   # (E, 3) cartesian bar vectors
    lattice: np.ndarray   # (3, 3) rows are lattice vectors

    @property
    def z_avg(self) -> float:
        return 2 * len(self.edges) / len(self.frac)


def build_periodic_graph(lat: Lattice, tol: float = 1e-4) -> PeriodicGraph:
    # Merge nodes that are periodic copies of each other (differ by an integer vector).
    unique: list[np.ndarray] = []
    node_map = np.empty(len(lat.nodes), dtype=int)
    for i, p in enumerate(lat.nodes):
        for k, q in enumerate(unique):
            d = p - q
            if np.all(np.abs(d - np.round(d)) < tol):
                node_map[i] = k
                break
        else:
            node_map[i] = len(unique)
            unique.append(p - np.floor(p + tol))   # wrap into [0, 1)
    frac = np.array(unique)
    shifts = np.round(lat.nodes - frac[node_map]).astype(int)

    # Each bar becomes (i, j, offset); store one canonical direction and drop duplicates.
    seen = set()
    edges, offsets = [], []
    for a, b in lat.bars:
        i, j = node_map[a], node_map[b]
        off = shifts[b] - shifts[a]
        if i == j and not off.any():
            continue   # zero-length bar
        if i > j or (i == j and tuple(off) < (0, 0, 0)):
            i, j, off = j, i, -off
        key = (i, j, *off.tolist())
        if key in seen:
            continue
        seen.add(key)
        edges.append((i, j))
        offsets.append(off)

    L = lattice_matrix(lat.cell)
    edges = np.array(edges, dtype=int).reshape(-1, 2)
    offsets = np.array(offsets, dtype=int).reshape(-1, 3)
    vectors = (frac[edges[:, 1]] + offsets - frac[edges[:, 0]]) @ L
    return PeriodicGraph(frac, frac @ L, edges, offsets, vectors, L)


def edge_features(vectors):
    """[length, n_x, n_y, n_z, n⊗n (xx, yy, zz, yz, xz, xy)] per bar, from a (E, 3) torch tensor.

    n⊗n is the same for both directions of a bar and is what a bar's axial
    stiffness contributes to the effective stiffness tensor.
    """
    import torch

    length = vectors.norm(dim=1, keepdim=True)
    n = vectors / length
    nn = torch.stack([n[:, 0]**2, n[:, 1]**2, n[:, 2]**2,
                      n[:, 1] * n[:, 2], n[:, 0] * n[:, 2], n[:, 0] * n[:, 1]], dim=1)
    return torch.cat([length, n, nn], dim=1)


def _moduli(props: dict) -> dict:
    return {"x": props["Ex"], "y": props["Ey"], "z": props["Ez"]}


def coupling(props: dict, i: str, j: str) -> float:
    """Symmetric, dimensionless Poisson coupling c_ij = -S_ij * sqrt(E_i E_j).

    The file gives both nu_ij and nu_ji (rounded to 3 decimals); in theory
    nu_ij * sqrt(E_j/E_i) == nu_ji * sqrt(E_i/E_j) == c_ij, so we average the two.
    """
    E = _moduli(props)
    return 0.5 * (props[f"nu{i}{j}"] * np.sqrt(E[j] / E[i])
                  + props[f"nu{j}{i}"] * np.sqrt(E[i] / E[j]))


def poisson_symmetry_error(props: dict) -> float:
    """Largest disagreement between the two estimates of c_ij (should be ~0)."""
    E = _moduli(props)
    return max(abs(props[f"nu{i}{j}"] * np.sqrt(E[j] / E[i])
                   - props[f"nu{j}{i}"] * np.sqrt(E[i] / E[j])) for i, j in PAIRS)


def target_vector(props: dict) -> np.ndarray:
    """9 independent orthotropic constants: log10 of the 6 moduli, then c_yz, c_xz, c_xy."""
    moduli = np.array([props[k] for k in MODULI])
    logs = np.log10(np.maximum(moduli, MIN_MODULUS))
    return np.concatenate([logs, [coupling(props, i, j) for i, j in PAIRS]])


def props_from_target(t: np.ndarray) -> dict:
    """Inverse of target_vector: back to Ex ... Gxy and all six Poisson's ratios."""
    props = {k: 10.0 ** t[n] for n, k in enumerate(MODULI)}
    E = _moduli(props)
    for (i, j), c in zip(PAIRS, t[6:]):
        props[f"nu{i}{j}"] = c * np.sqrt(E[i] / E[j])
        props[f"nu{j}{i}"] = c * np.sqrt(E[j] / E[i])
    return props


def compliance_matrix(props: dict) -> np.ndarray:
    """6x6 orthotropic compliance in Voigt order (xx, yy, zz, yz, xz, xy), engineering shear strains."""
    E = _moduli(props)
    axis = {"x": 0, "y": 1, "z": 2}
    S = np.zeros((6, 6))
    for n, k in enumerate(MODULI):
        S[n, n] = 1.0 / props[k]
    for i, j in PAIRS:
        s = -coupling(props, i, j) / np.sqrt(E[i] * E[j])
        S[axis[i], axis[j]] = S[axis[j], axis[i]] = s
    return S


def stiffness_matrix(props: dict) -> np.ndarray:
    """6x6 stiffness C = S^-1. Can be ill-conditioned for near-incompressible lattices (nu ~ 0.5)."""
    return np.linalg.inv(compliance_matrix(props))


def to_pyg(lat: Lattice, tol: float = 1e-4, with_targets: bool = True):
    """Build a torch_geometric Data object for one lattice.

    x          (N, 1)    node coordination number
    edge_index (2, 2E)   both directions of every bar
    edge_vec   (2E, 3)   cartesian bar vectors (used to redo edge_attr after augmentation)
    edge_attr  (2E, 10)  see edge_features
    u          (1, 2)    graph-level: [Z_avg, log10(total bar length / cell volume)]
    y          (1, 9)    see TARGET_NAMES             } only if with_targets and the
    C          (1, 6, 6) orthotropic stiffness matrix } file gives properties
    """
    import torch
    from torch_geometric.data import Data

    g = build_periodic_graph(lat, tol)
    src = np.concatenate([g.edges[:, 0], g.edges[:, 1]])
    dst = np.concatenate([g.edges[:, 1], g.edges[:, 0]])
    vec = torch.tensor(np.concatenate([g.vectors, -g.vectors]), dtype=torch.float32)
    degree = np.bincount(src, minlength=len(g.frac))

    volume = abs(np.linalg.det(g.lattice))
    total_length = np.linalg.norm(g.vectors, axis=1).sum()

    data = Data(
        x=torch.tensor(degree[:, None], dtype=torch.float32),
        edge_index=torch.tensor(np.stack([src, dst]), dtype=torch.long),
        edge_vec=vec,
        edge_attr=edge_features(vec),
        u=torch.tensor([[g.z_avg, np.log10(total_length / volume)]], dtype=torch.float32),
        name=lat.name,
    )
    if with_targets and lat.props is not None:
        data.y = torch.tensor(target_vector(lat.props)[None], dtype=torch.float32)
        data.C = torch.tensor(stiffness_matrix(lat.props)[None], dtype=torch.float64)
    return data
