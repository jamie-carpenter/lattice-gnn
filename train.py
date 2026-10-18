"""Train the lattice GNN.

Usage:
    python train.py data/raw/lattices.txt
    python train.py data/raw/lattices.txt --epochs 300 --split group

Saves the model with the lowest validation loss to checkpoints/model.pt, and the
train/val/test split and test-set results next to it (split.json, metrics.json).

Splits:
  random  each lattice goes to train, val or test at random (default). Every test
          lattice is one the model has never seen.
  group   whole families (same symmetry and connectivity, e.g. all cub_Z08.0_*)
          are held out together - a harder test of generalizing to new kinds of lattice.
"""
import argparse
import json
import math
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from lattice_gnn.augment import random_axis_transform
from lattice_gnn.data import load_dataset
from lattice_gnn.graph import DISPLAY_NAMES
from lattice_gnn.model import LatticeGNN

N_MODULI = 6   # the first 6 targets are log10 moduli, the last 3 Poisson couplings


def family(name: str) -> str:
    """Group key for --split group, e.g. cub_Z08.0_E3 -> cub_Z08.0."""
    return re.sub(r"_E\d+$", "", name)


def split_dataset(graphs, val_frac, test_frac, mode, seed):
    if len(graphs) < 10:
        print(f"WARNING: only {len(graphs)} lattices, too few to split. Training and evaluating on "
              "the same lattices: this only checks that the code runs and the model can learn.\n")
        return graphs, graphs, graphs
    rng = random.Random(seed)
    if mode == "random":
        graphs = graphs[:]
        rng.shuffle(graphs)
        n_test, n_val = round(test_frac * len(graphs)), round(val_frac * len(graphs))
        return graphs[n_test + n_val:], graphs[n_test:n_test + n_val], graphs[:n_test]

    groups = {}
    for g in graphs:
        groups.setdefault(family(g.name), []).append(g)
    keys = sorted(groups)
    rng.shuffle(keys)
    train, val, test = [], [], []
    for key in keys:
        if len(test) < test_frac * len(graphs):
            test += groups[key]
        elif len(val) < val_frac * len(graphs):
            val += groups[key]
        else:
            train += groups[key]
    return train, val, test


@torch.no_grad()
def evaluate(model, loader):
    """Mean standardized MSE, predictions and true values (in target units)."""
    model.eval()
    sq_err, preds, trues = 0.0, [], []
    for batch in loader:
        out = model(batch)
        sq_err += F.mse_loss(out, model.y_norm(batch.y), reduction="sum").item()
        preds.append(model.y_norm.inverse(out))
        trues.append(batch.y)
    pred, true = torch.cat(preds), torch.cat(trues)
    return sq_err / true.numel(), pred, true


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


def main():
    parser = argparse.ArgumentParser(description="Train the lattice GNN.")
    parser.add_argument("paths", nargs="+", help="lattice data file(s) with properties")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128, help="width of the network")
    parser.add_argument("--layers", type=int, default=4, help="number of message-passing layers")
    parser.add_argument("--split", choices=["random", "group"], default="random")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=40,
                        help="stop if validation loss hasn't improved for this many epochs")
    parser.add_argument("--no-augment", action="store_true", help="turn off axis-swap augmentation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rebuild", action="store_true", help="re-parse data files, ignoring the cache")
    parser.add_argument("--out", default="checkpoints/model.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    graphs = load_dataset(args.paths, rebuild=args.rebuild)
    train, val, test = split_dataset(graphs, args.val_frac, args.test_frac, args.split, args.seed)
    print(f"{len(graphs)} lattices: {len(train)} train, {len(val)} val, {len(test)} test")

    first = train[0]
    model = LatticeGNN(node_dim=first.x.size(1), edge_dim=first.edge_attr.size(1),
                       global_dim=first.u.size(1), out_dim=first.y.size(1),
                       hidden=args.hidden, layers=args.layers)
    model.fit_normalizers(train)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,
                                                           patience=max(args.patience // 4, 1))
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=256)
    test_loader = DataLoader(test, batch_size=256)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best_loss, best_epoch = math.inf, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for batch in train_loader:
            if not args.no_augment:
                batch = random_axis_transform(batch)
            loss = F.mse_loss(model(batch), model.y_norm(batch.y))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * batch.num_graphs
            count += batch.num_graphs

        val_loss, _, _ = evaluate(model, val_loader)
        scheduler.step(val_loss)
        improved = val_loss < best_loss
        if improved:
            best_loss, best_epoch = val_loss, epoch
            model.save(out, args=vars(args), epoch=epoch)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:4d}  train loss {total / count:.4f}  val loss {val_loss:.4f}"
                  f"  lr {optimizer.param_groups[0]['lr']:.1e}" + ("  (best)" if improved else ""))
        if epoch - best_epoch >= args.patience:
            print(f"Stopping: no improvement for {args.patience} epochs.")
            break

    model, _ = LatticeGNN.load(out)
    test_loss, pred, true = evaluate(model, test_loader)
    results = metrics(pred, true)
    print(f"\nBest model from epoch {best_epoch}, saved to {out}")
    print(f"Test set ({len(test)} lattices), loss {test_loss:.4f}:\n")
    print_metrics(results)

    (out.parent / "metrics.json").write_text(json.dumps(
        {"best_epoch": best_epoch, "test_loss": test_loss, "test": results}, indent=2))
    (out.parent / "split.json").write_text(json.dumps(
        {name: [g.name for g in part] for name, part in
         (("train", train), ("val", val), ("test", test))}, indent=2))


if __name__ == "__main__":
    main()
