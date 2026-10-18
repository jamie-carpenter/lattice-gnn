"""Data augmentation by relabeling and mirroring the x, y, z axes.

Swapping two axes (e.g. x <-> y) or mirroring one is the same lattice seen from a
different direction, so its true properties are known exactly: swapping x and y
swaps Ex with Ey, Gxz with Gyz, and so on. Mirroring leaves all 9 targets unchanged.
Training on these 48 views teaches the model this symmetry instead of memorizing
one orientation.
"""
import itertools

import torch

from .graph import edge_features

PERMUTATIONS = list(itertools.permutations(range(3)))


def transform_axes(batch, perm, signs):
    """New axis k is old axis perm[k], multiplied by signs[k] (+1 or -1). Returns a copy.

    Target layout (see graph.TARGET_NAMES): [Ex, Ey, Ez], then G and Poisson
    couplings indexed by the axis they *don't* involve (Gyz <-> x, Gxz <-> y,
    Gxy <-> z), so all three groups permute the same way.
    """
    batch = batch.clone()
    p = torch.as_tensor(perm)
    batch.edge_vec = batch.edge_vec[:, p] * torch.as_tensor(signs, dtype=batch.edge_vec.dtype)
    batch.edge_attr = edge_features(batch.edge_vec)
    if batch.y is not None:
        batch.y = batch.y[:, torch.cat([p, p + 3, p + 6])]
    if "C" in batch:
        voigt = torch.cat([p, p + 3])
        batch.C = batch.C[:, voigt][:, :, voigt]
    return batch


def random_axis_transform(batch):
    perm = PERMUTATIONS[torch.randint(len(PERMUTATIONS), ()).item()]
    signs = torch.randint(0, 2, (3,)) * 2 - 1
    return transform_axes(batch, perm, signs)
