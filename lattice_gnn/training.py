"""Training as a function: train(config) -> run_id. train.py is the command-line front end.

Every run writes its own folder runs/<id>/:
  config.json   the full config, with the seed and git commit
  split.json    the train/val/test split used
  model.pt      the model with the lowest validation loss
  history.json  loss per epoch
  eval.json     validation report (lattice_gnn.evaluate); the test set is never touched here
"""
from __future__ import annotations

import json
import math
import random
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from .augment import random_axis_transform
from .data import load_dataset
from .evaluate import evaluate
from .model import LatticeGNN
from .splits import apply_split, family_split, load_split, random_split, save_split

DEFAULTS = {
    "paths": [],             # lattice data file(s) with properties
    "split_file": None,      # frozen split (splits/<name>.json); if None, `split` is made per run
    "split": "random",       # random | group, only used without split_file
    "val_frac": 0.1,
    "test_frac": 0.1,
    "epochs": 200,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "hidden": 128,           # width of the network
    "layers": 4,             # number of message-passing layers
    "patience": 40,          # stop if validation loss hasn't improved for this many epochs
    "augment": True,         # axis-swap augmentation
    "pd_head": False,        # couplings always give a positive definite stiffness (model.pd_couplings)
    "seed": 0,
    "runs_dir": "runs",
    "run_id": None,          # default: timestamp + random suffix
    "rebuild": False,        # re-parse data files, ignoring the cache
    "verbose": True,
}


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_split(config: dict, graphs) -> dict:
    if config["split_file"]:
        return load_split(config["split_file"])
    make = random_split if config["split"] == "random" else family_split
    return make([g.name for g in graphs], config["val_frac"], config["test_frac"], config["seed"])


def train(config: dict, graphs=None) -> str:
    """Train one model and write runs/<id>/. `graphs` skips reloading the data (e.g. in a study)."""
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    config = {**DEFAULTS, **config}
    run_id = config["run_id"] or f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    run = Path(config["runs_dir"]) / run_id
    run.mkdir(parents=True, exist_ok=False)
    log = print if config["verbose"] else (lambda *a, **k: None)

    seed_everything(config["seed"])
    if graphs is None:
        graphs = load_dataset(config["paths"], rebuild=config["rebuild"])
    split = make_split(config, graphs)
    save_split(split, run / "split.json")
    if len(graphs) < 10 and not config["split_file"]:
        log(f"WARNING: only {len(graphs)} lattices, too few to split. Training and evaluating on "
            "the same lattices: this only checks that the code runs and the model can learn.\n")
        train_set = val_set = test_set = graphs
    else:
        train_set, val_set, test_set = apply_split(graphs, split)
    if not train_set or not val_set:
        raise ValueError(f"empty train or val set ({len(train_set)} train, {len(val_set)} val)")
    log(f"{len(graphs)} lattices: {len(train_set)} train, {len(val_set)} val, "
        f"{len(test_set)} test (not used)")
    (run / "config.json").write_text(json.dumps(
        {**config, "run_id": run_id, "git_commit": git_commit(), "torch": torch.__version__}, indent=1))

    first = train_set[0]
    model = LatticeGNN(node_dim=first.x.size(1), edge_dim=first.edge_attr.size(1),
                       global_dim=first.u.size(1), out_dim=first.y.size(1),
                       hidden=config["hidden"], layers=config["layers"], pd_head=config["pd_head"])
    model.fit_normalizers(train_set)
    log(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,
                                                           patience=max(config["patience"] // 4, 1))
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True,
                              generator=torch.Generator().manual_seed(config["seed"]))
    val_loader = DataLoader(val_set, batch_size=256)

    history = []
    best_loss, best_epoch = math.inf, 0
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total, count = 0.0, 0
        for batch in train_loader:
            if config["augment"]:
                batch = random_axis_transform(batch)
            loss = F.mse_loss(model(batch), model.y_norm(batch.y))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * batch.num_graphs
            count += batch.num_graphs

        val_loss = validation_loss(model, val_loader)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": total / count, "val_loss": val_loss, "lr": lr})
        improved = val_loss < best_loss
        if improved:
            best_loss, best_epoch = val_loss, epoch
            model.save(run / "model.pt", args=config, epoch=epoch)
        if epoch == 1 or epoch % 10 == 0 or epoch == config["epochs"]:
            log(f"epoch {epoch:4d}  train loss {total / count:.4f}  val loss {val_loss:.4f}"
                f"  lr {lr:.1e}" + ("  (best)" if improved else ""))
        if epoch - best_epoch >= config["patience"]:
            log(f"Stopping: no improvement for {config['patience']} epochs.")
            break
    (run / "history.json").write_text(json.dumps({"best_epoch": best_epoch, "epochs": history}, indent=1))

    model, _ = LatticeGNN.load(run / "model.pt")
    result = evaluate(model, val_set, "val")
    result["best_epoch"] = best_epoch
    (run / "eval.json").write_text(json.dumps(result, indent=1))
    log(f"\nBest model from epoch {best_epoch}, saved to {run}")
    return run_id


@torch.no_grad()
def validation_loss(model, loader) -> float:
    """Mean standardized MSE, the same quantity the training loss measures."""
    model.eval()
    sq_err, n = 0.0, 0
    for batch in loader:
        sq_err += F.mse_loss(model(batch), model.y_norm(batch.y), reduction="sum").item()
        n += batch.y.numel()
    return sq_err / n
