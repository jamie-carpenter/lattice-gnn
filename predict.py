"""Predict the stiffness of lattices with a trained model.

The input file uses the same format as the training data; the mechanical
properties can be left out (unknown lattices). If they are present, the
predictions are printed next to the true values.

Usage:
    python predict.py data/new_lattices.txt
    python predict.py data/new_lattices.txt --checkpoint checkpoints/model.pt --out predictions.csv

Output CSV columns: name, Ex ... Gxy, the six Poisson's ratios, and C11 ... C66,
the 21 independent entries of the 6x6 stiffness matrix (Voigt order xx, yy, zz, yz, xz, xy).
"""
import argparse
import csv
import warnings

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from lattice_gnn.graph import props_from_target, stiffness_matrix, to_pyg
from lattice_gnn.model import LatticeGNN
from lattice_gnn.parse import PROPERTY_KEYS, read_lattices

UPPER = [(i, j) for i in range(6) for j in range(i, 6)]
SHOW = 10   # lattices to print when comparing with true values


def main():
    parser = argparse.ArgumentParser(description="Predict lattice stiffness with a trained model.")
    parser.add_argument("path", help="lattice file (properties optional)")
    parser.add_argument("--checkpoint", default="checkpoints/model.pt")
    parser.add_argument("--out", default="predictions.csv")
    args = parser.parse_args()

    model, _ = LatticeGNN.load(args.checkpoint)
    lattices, graphs = [], []
    for lat in read_lattices(args.path, strict=False):
        try:
            graphs.append(to_pyg(lat, with_targets=False))
            lattices.append(lat)
        except Exception as err:
            warnings.warn(f"{lat.name}: could not build graph ({err}), skipped")

    loader = DataLoader(graphs, batch_size=256)
    targets = torch.cat([model.predict(batch) for batch in loader]).double().numpy()

    rows, not_pd = [], 0
    for lat, t in zip(lattices, targets):
        props = props_from_target(t)
        C = stiffness_matrix(props)
        not_pd += np.linalg.eigvalsh(C).min() <= 0
        rows.append([lat.name] + [props[k] for k in PROPERTY_KEYS] + [C[i, j] for i, j in UPPER])

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", *PROPERTY_KEYS, *(f"C{i + 1}{j + 1}" for i, j in UPPER)])
        writer.writerows(rows)
    print(f"Wrote predictions for {len(rows)} lattices to {args.out}")
    if not_pd:
        print(f"WARNING: {not_pd} predicted stiffness matrices are not positive definite "
              "(physically invalid); treat those predictions with caution.")

    known = [(lat, row) for lat, row in zip(lattices, rows) if lat.props is not None]
    if known:
        print(f"\nPredicted vs true (first {min(SHOW, len(known))} of {len(known)} with known values):")
        for lat, row in known[:SHOW]:
            print(f"\n{lat.name}")
            for k, predicted in zip(PROPERTY_KEYS, row[1:]):
                fmt = ".2E" if k[0] in "EG" else ".3f"
                print(f"  {k:<5} true {lat.props[k]:{fmt}}   predicted {predicted:{fmt}}")


if __name__ == "__main__":
    main()
