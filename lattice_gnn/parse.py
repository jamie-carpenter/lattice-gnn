"""Read lattice records from the text dataset format.

Each record looks like:

    Name: cub_Z06.0_E1
    Other name(s): ...
    Normalized unit cell parameters (a,b,c,alpha,beta,gamma):
    1.00000, 1.00000, 1.00000, 90.0, 90.0, 90.0
    Average connectivity: Z_avg = 6.00
    Effective normalized mechanical properties:
    Ex  = 3.34E-03, Ey  = 3.34E-03, Ez  = 3.34E-03
    ...
    Nodal positions:
    0.00000 0.00000 0.00000
    ...
    Bar connectivities:
    1 5
    ...

Records are separated by a line of dashes. Node positions are fractional
coordinates in the unit cell and bar indices are 1-based in the file.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FLOAT = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
SEPARATOR = re.compile(r"^\s*-{10,}\s*$", re.MULTILINE)

PROPERTY_KEYS = ("Ex", "Ey", "Ez", "Gyz", "Gxz", "Gxy",
                 "nuyz", "nuxz", "nuxy", "nuzy", "nuzx", "nuyx")


@dataclass
class Lattice:
    """One lattice. Properties are None when the file doesn't give them (unknown lattices)."""
    name: str
    other_names: list[str]
    cell: np.ndarray                 # (6,) a, b, c, alpha, beta, gamma (degrees)
    z_avg: float | None              # average connectivity reported in the file
    props: dict[str, float] | None   # Ex ... nuyx, see PROPERTY_KEYS
    scaling_C: np.ndarray | None     # (3,) Cx, Cy, Cz  (E_i ~ C_i * rho^n_i)
    scaling_n: np.ndarray | None     # (3,) nx, ny, nz
    nodes: np.ndarray         # (N, 3) fractional coordinates
    bars: np.ndarray          # (M, 2) zero-based node indices
    e_max: float | None       # only reported when max stiffness is off-axis
    intersecting: bool        # file flags overlapping/intersecting bars


def _find(pattern: str, text: str, where: str, flags: int = re.MULTILINE) -> re.Match:
    m = re.search(pattern, text, flags)
    if m is None:
        raise ValueError(f"{where}: could not find {pattern!r}")
    return m


def _rows(section: str, ncols: int, where: str) -> np.ndarray:
    rows = [[float(v) for v in line.split()]
            for line in section.strip().splitlines() if line.strip()]
    arr = np.array(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != ncols:
        raise ValueError(f"{where}: expected {ncols} columns, got shape {arr.shape}")
    return arr


def parse_record(text: str) -> Lattice:
    name = _find(r"^Name:\s*(\S+)", text, "record").group(1)

    other = re.search(r"^Other name\(s\):\s*(.*)$", text, re.MULTILINE)
    other_names = [s.strip() for s in other.group(1).split(",") if s.strip()] if other else []

    cell_line = _find(r"unit cell parameters[^\n]*\n\s*([^\n]+)", text, name).group(1)
    cell = np.array([float(v) for v in cell_line.split(",")])

    def value(key: str) -> float:
        return float(_find(rf"\b{key}\s*=\s*({FLOAT})", text, name).group(1))

    def has(key: str) -> bool:
        return re.search(rf"\b{key}\s*=", text) is not None

    e_max = re.search(rf"\bEmax\s*=\s*({FLOAT})", text)

    nodes = _find(r"Nodal positions:\s*\n(.*?)\n\s*Bar connectivities:",
                  text, name, re.DOTALL).group(1)
    bars = _find(r"Bar connectivities:\s*\n(.*)", text, name, re.DOTALL).group(1)

    return Lattice(
        name=name,
        other_names=other_names,
        cell=cell,
        z_avg=value("Z_avg") if has("Z_avg") else None,
        props={k: value(k) for k in PROPERTY_KEYS} if has("Ex") else None,
        scaling_C=np.array([value(k) for k in ("Cx", "Cy", "Cz")]) if has("Cx") else None,
        scaling_n=np.array([value(k) for k in ("nx", "ny", "nz")]) if has("nx") else None,
        nodes=_rows(nodes, 3, name),
        bars=_rows(bars, 2, name).astype(int) - 1,
        e_max=float(e_max.group(1)) if e_max else None,
        intersecting="overlapping or intersecting bars" in text,
    )


def read_lattices(path: str | Path, strict: bool = True) -> list[Lattice]:
    """Parse every record in a file. With strict=False, bad records are skipped with a warning."""
    text = Path(path).read_text(encoding="utf-8-sig")   # -sig: ignore a BOM some Windows tools add
    lattices = []
    for block in SEPARATOR.split(text):
        if "Name:" not in block:
            continue
        try:
            lattices.append(parse_record(block))
        except ValueError as err:
            if strict:
                raise
            warnings.warn(f"skipping record: {err}")
    return lattices
