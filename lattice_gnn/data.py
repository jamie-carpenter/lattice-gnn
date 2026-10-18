"""Load lattice files as lists of graphs, caching the result so parsing happens once."""
from __future__ import annotations

import warnings
from pathlib import Path

import torch

from .graph import to_pyg
from .parse import read_lattices

# Bump this when to_pyg changes, so old cached graphs aren't reused.
CACHE_VERSION = 1


def build_graphs(path: str | Path) -> list:
    """Graphs for every lattice in a file that has properties (i.e. can be trained on)."""
    graphs = []
    for lat in read_lattices(path, strict=False):
        if lat.props is None:
            warnings.warn(f"{lat.name}: no properties in file, skipped")
            continue
        try:
            graphs.append(to_pyg(lat))
        except Exception as err:
            warnings.warn(f"{lat.name}: could not build graph ({err}), skipped")
    return graphs


def load_dataset(paths, cache_dir: str | Path = "data/processed", rebuild: bool = False) -> list:
    graphs = []
    for path in map(Path, paths):
        cache = Path(cache_dir) / f"{path.stem}.v{CACHE_VERSION}.pt"
        if not rebuild and cache.exists() and cache.stat().st_mtime > path.stat().st_mtime:
            graphs += torch.load(cache, weights_only=False)
            continue
        built = build_graphs(path)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(built, cache)
        graphs += built
    return graphs
