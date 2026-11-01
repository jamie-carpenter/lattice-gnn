"""Sanity-check a lattice data file before training.

For every record: parse it, build the periodic graph, and check that
  - the graph's average connectivity matches the Z_avg reported in the file (computed
    with the catalog's convention, PeriodicGraph.catalog_z_avg),
  - the Poisson's ratios satisfy nu_ij / E_i == nu_ji / E_j (up to rounding),
  - the resulting stiffness matrix is positive definite.

Usage:
    python check_data.py data/samples/sample_lattices.txt
"""
import argparse

import numpy as np

from lattice_gnn.graph import (build_periodic_graph, poisson_symmetry_error, props_from_target,
                               stiffness_matrix, target_vector, MODULI)
from lattice_gnn.parse import read_lattices

NU_TOL = 0.02
POISSON = ("nuyz", "nuxz", "nuxy", "nuzy", "nuzx", "nuyx")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", help="lattice data file")
    parser.add_argument("--show", type=int, default=20, help="print details for the first N records")
    args = parser.parse_args()

    lattices = read_lattices(args.path, strict=False)
    print(f"Parsed {len(lattices)} lattices from {args.path}\n")

    problems = {"Z_avg mismatch": [], "Poisson asymmetry": [], "C not positive definite": [],
                "No properties (can't train on)": []}
    for n, lat in enumerate(lattices):
        g = build_periodic_graph(lat)
        z_ok = lat.z_avg is None or abs(g.catalog_z_avg() - lat.z_avg) < 0.01
        if not z_ok:
            problems["Z_avg mismatch"].append(lat.name)
        if lat.props is None:
            problems["No properties (can't train on)"].append(lat.name)
        else:
            if poisson_symmetry_error(lat.props) > NU_TOL:
                problems["Poisson asymmetry"].append(lat.name)
            if np.linalg.eigvalsh(stiffness_matrix(lat.props)).min() <= 0:
                problems["C not positive definite"].append(lat.name)

        if n < args.show:
            print(f"{lat.name}")
            print(f"  nodes {len(lat.nodes)} -> {len(g.frac)} unique, "
                  f"bars {len(lat.bars)} -> {len(g.edges)} unique")
            file_z = "not given" if lat.z_avg is None else f"{lat.z_avg:.2f}"
            print(f"  Z_avg graph {g.catalog_z_avg():.2f} (true average degree {g.z_avg:.2f}), "
                  f"file {file_z}  {'OK' if z_ok else 'MISMATCH'}")
            if lat.props is not None:
                # Round-trip through the training targets, so this shows what the model's
                # predictions look like once converted back to the file's properties.
                props = props_from_target(target_vector(lat.props))
                print("  " + ", ".join(f"{k}={props[k]:.2E}" for k in MODULI))
                print("  " + ", ".join(f"{k}={props[k]:.3f}" for k in POISSON))
            print()

    print("Summary")
    for label, names in problems.items():
        print(f"  {label}: {len(names)}" + (f"  e.g. {', '.join(names[:5])}" if names else ""))


if __name__ == "__main__":
    main()
