"""Evaluate a trained model: errors per target and per family, and a positive-definiteness check.

The report (a dict, saved as runs/<id>/eval.json) is what the optimizer and the agent read.

Usage:
    python -m lattice_gnn.evaluate runs/<id>                # validation set -> eval.json
    python -m lattice_gnn.evaluate runs/<id> --part test    # locked test set -> eval_test.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from .graph import DISPLAY_NAMES, PAIRS
from .splits import family

N_MODULI = 6   # the first 6 targets are log10 moduli, the last 3 Poisson couplings
WORST = 10     # families listed in "worst_families"

# Positive-definiteness threshold.
# The orthotropic compliance S has positive diagonal 1/E_i (the moduli are 10^prediction > 0)
# and S_ij = -c_ij / sqrt(E_i E_j) off the diagonal. Scaling S by D = diag(sqrt(E)) on both
# sides keeps its definiteness (Sylvester's law of inertia), and gives a 3x3 normal block with
# 1 on the diagonal and -c_ij off it, plus an identity shear block. So S (and C = S^-1) is
# positive definite exactly when that 3x3 block is, which depends only on the 3 couplings and
# avoids inverting S (moduli span ~5 orders of magnitude). The block's trace is 3, so its
# eigenvalues are at most 3. Predictions are float32, so each coupling carries a rounding error
# up to eps32 * |c| <= eps32 (|c| < 1 for any positive definite block), and by Weyl's inequality
# an eigenvalue moves by at most the spectral norm of that error matrix, < 3 * eps32.
# Eigenvalues below that cannot be told apart from 0, so positive definite means > PD_TOL.
PD_TOL = 3 * float(np.finfo(np.float32).eps)


def normalized_block(couplings: np.ndarray) -> np.ndarray:
    """(n, 3) couplings [c_yz, c_xz, c_xy] -> (n, 3, 3) normalized compliance blocks."""
    axis = {"x": 0, "y": 1, "z": 2}
    blocks = np.tile(np.eye(3), (len(couplings), 1, 1))
    for k, (i, j) in enumerate(PAIRS):
        blocks[:, axis[i], axis[j]] = blocks[:, axis[j], axis[i]] = -couplings[:, k]
    return blocks


def is_positive_definite(targets: np.ndarray) -> np.ndarray:
    """(n, 9) targets -> (n,) bool: is the predicted 6x6 stiffness positive definite?"""
    eig = np.linalg.eigvalsh(normalized_block(np.asarray(targets, dtype=np.float64)[:, N_MODULI:]))
    return eig[:, 0] > PD_TOL


@torch.no_grad()
def predict(model, graphs, batch_size: int = 256):
    """Predictions and true values in target units, plus the standardized MSE (the training loss)."""
    model.eval()
    preds, trues = [], []
    for batch in DataLoader(graphs, batch_size=batch_size):
        preds.append(model.predict(batch))
        trues.append(batch.y)
    pred, true = torch.cat(preds), torch.cat(trues)
    loss = (((pred - true) / model.y_norm.std) ** 2).mean().item()
    return pred, true, loss


def metrics(pred, true) -> dict:
    """Per property: R² and a typical error.

    Moduli are predicted as log10, so their error is reported as a median
    percentage (10^|error| - 1). Poisson couplings get a median absolute error.
    """
    results = {}
    for k, name in enumerate(DISPLAY_NAMES):
        diff = pred[:, k] - true[:, k]
        ss_tot = ((true[:, k] - true[:, k].mean()) ** 2).sum()
        r2 = (1 - (diff ** 2).sum() / ss_tot).item() if ss_tot > 0 else math.nan
        if k < N_MODULI:
            error = {"median_percent_error": ((10 ** diff.abs() - 1) * 100).median().item()}
        else:
            error = {"median_abs_error": diff.abs().median().item()}
        results[name] = {"R2": r2, **error}
    return results


def print_metrics(results: dict):
    print(f"{'Property':<12}{'R²':>8}   typical error")
    for name, m in results.items():
        error = (f"{m['median_percent_error']:.1f} %" if "median_percent_error" in m
                 else f"{m['median_abs_error']:.3f}")
        print(f"{name:<12}{m['R2']:>8.3f}   {error}")


def _errors(diff: np.ndarray) -> dict:
    """Median % error over all moduli and median abs error over all couplings of some lattices."""
    return {"median_percent_error_moduli": float(np.median((10 ** np.abs(diff[:, :N_MODULI]) - 1) * 100)),
            "median_abs_error_couplings": float(np.median(np.abs(diff[:, N_MODULI:])))}


def report(pred, true, names, part: str, loss: float | None = None) -> dict:
    """The evaluation report for one set of lattices (see module docstring)."""
    pred, true = torch.as_tensor(pred), torch.as_tensor(true)
    diff = (pred - true).double().numpy()
    pd_ok = is_positive_definite(pred.numpy())
    fams = np.array([family(n) for n in names])

    families = {}
    for fam in sorted(set(fams)):
        rows = fams == fam
        families[fam] = {"n": int(rows.sum()), **_errors(diff[rows]), "non_pd": int((~pd_ok[rows]).sum())}
    worst = sorted(families, key=lambda f: families[f]["median_percent_error_moduli"], reverse=True)

    return {
        "part": part,
        "n_lattices": len(names),
        "n_families": len(families),
        "loss": loss,
        "overall": _errors(diff),
        "targets": metrics(pred, true),
        "pd": {"threshold": PD_TOL, "non_pd": int((~pd_ok).sum()),
               "non_pd_rate": float((~pd_ok).mean()),
               "true_non_pd": int((~is_positive_definite(true.numpy())).sum())},
        "worst_families": worst[:WORST],
        "families": families,
    }


def evaluate(model, graphs, part: str) -> dict:
    pred, true, loss = predict(model, graphs)
    return report(pred, true, [g.name for g in graphs], part, loss)


def main():
    from .data import load_dataset
    from .model import LatticeGNN
    from .splits import apply_split, load_split

    parser = argparse.ArgumentParser(description="Evaluate a training run.")
    parser.add_argument("run", help="run folder, e.g. runs/<id>")
    parser.add_argument("--part", choices=["val", "test"], default="val")
    args = parser.parse_args()

    run = Path(args.run)
    config = json.loads((run / "config.json").read_text())
    model, _ = LatticeGNN.load(run / "model.pt")
    graphs = load_dataset(config["paths"])
    _, val, test = apply_split(graphs, load_split(run / "split.json"))
    result = evaluate(model, val if args.part == "val" else test, args.part)
    out = run / ("eval.json" if args.part == "val" else "eval_test.json")
    out.write_text(json.dumps(result, indent=1))
    print(f"{args.part} set: {result['n_lattices']} lattices, {result['n_families']} families, "
          f"loss {result['loss']:.4f}, non-PD {result['pd']['non_pd']}\n")
    print_metrics(result["targets"])
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
