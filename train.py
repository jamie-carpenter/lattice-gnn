"""Train the lattice GNN (command-line front end to lattice_gnn.training.train).

Usage:
    python train.py data/raw/lattices.txt --split-file splits/main.json
    python train.py data/raw/lattices.txt --epochs 300 --split group

Each run writes runs/<id>/ (config, split, best model, loss history, validation report);
see lattice_gnn/training.py. The test set is not evaluated here. Once, at the end:
    python -m lattice_gnn.evaluate runs/<id> --part test

Splits:
  --split-file  a frozen split from `python -m lattice_gnn.splits` (use this for real runs)
  random        each lattice goes to train, val or test at random (default without a split file)
  group         whole families (same symmetry and connectivity, e.g. all cub_Z08.0_*)
                are held out together - a harder test of generalizing to new kinds of lattice
"""
import argparse
import json
from pathlib import Path

from lattice_gnn.evaluate import print_metrics
from lattice_gnn.training import DEFAULTS, train


def main():
    d = DEFAULTS
    parser = argparse.ArgumentParser(description="Train the lattice GNN.")
    parser.add_argument("paths", nargs="+", help="lattice data file(s) with properties")
    parser.add_argument("--split-file", default=None, help="frozen split, e.g. splits/main.json")
    parser.add_argument("--split", choices=["random", "group"], default=d["split"])
    parser.add_argument("--val-frac", type=float, default=d["val_frac"])
    parser.add_argument("--test-frac", type=float, default=d["test_frac"])
    parser.add_argument("--epochs", type=int, default=d["epochs"])
    parser.add_argument("--batch-size", type=int, default=d["batch_size"])
    parser.add_argument("--lr", type=float, default=d["lr"])
    parser.add_argument("--weight-decay", type=float, default=d["weight_decay"])
    parser.add_argument("--hidden", type=int, default=d["hidden"], help="width of the network")
    parser.add_argument("--layers", type=int, default=d["layers"],
                        help="number of message-passing layers")
    parser.add_argument("--patience", type=int, default=d["patience"],
                        help="stop if validation loss hasn't improved for this many epochs")
    parser.add_argument("--no-augment", action="store_true", help="turn off axis-swap augmentation")
    parser.add_argument("--pd-head", action="store_true",
                        help="predict couplings that always give a positive definite stiffness")
    parser.add_argument("--seed", type=int, default=d["seed"])
    parser.add_argument("--rebuild", action="store_true", help="re-parse data files, ignoring the cache")
    parser.add_argument("--runs-dir", default=d["runs_dir"])
    parser.add_argument("--run-id", default=None)
    args = vars(parser.parse_args())

    args["augment"] = not args.pop("no_augment")
    run_id = train(args)

    run = Path(args["runs_dir"]) / run_id
    result = json.loads((run / "eval.json").read_text())
    print(f"Validation set ({result['n_lattices']} lattices, {result['n_families']} families), "
          f"loss {result['loss']:.4f}, non-PD predictions {result['pd']['non_pd']}:\n")
    print_metrics(result["targets"])


if __name__ == "__main__":
    main()
