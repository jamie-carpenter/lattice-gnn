"""Splits, evaluation, train(config) and the Optuna baseline, on the committed sample file."""
import json

import numpy as np
import pytest
import torch

from lattice_gnn.data import load_dataset
from lattice_gnn.evaluate import evaluate, is_positive_definite, report
from lattice_gnn.graph import props_from_target, stiffness_matrix
from lattice_gnn.model import LatticeGNN
from lattice_gnn.optimize import check_target, optimize
from lattice_gnn.splits import apply_split, family, family_split, load_split, save_split
from lattice_gnn.training import train

SAMPLE = "data/samples/sample_lattices.txt"


@pytest.fixture(scope="module")
def graphs(tmp_path_factory):
    return load_dataset([SAMPLE], cache_dir=tmp_path_factory.mktemp("cache"))


def test_family():
    assert family("cub_Z08.0_E3") == family("cub_Z08.0_R12") == "cub_Z08.0"


def test_family_split_keeps_families_and_duplicates_together(tmp_path):
    names = [f"{s}_Z0{z}.0_E{i}" for s in ("cub", "tet", "ort") for z in range(3, 9) for i in range(5)]
    names.append("cub_Z09.0_R1")
    split = family_split(names, 0.2, 0.2, seed=1, duplicates=[("cub_Z09.0_R1", "tet_Z03.0_E0")])
    parts = [p for p in ("train", "val", "test") for f in split[p]]
    assert len(parts) == len({family(n) for n in names})          # each family in one part
    part_of = {f: p for p in ("train", "val", "test") for f in split[p]}
    assert part_of["cub_Z09.0"] == part_of["tet_Z03.0"]
    save_split(split, tmp_path / "s.json")
    assert load_split(tmp_path / "s.json") == split


def test_apply_split_rejects_unknown(graphs):
    with pytest.raises(ValueError):
        apply_split(graphs, {"by": "family", "train": [], "val": [], "test": []})


def test_pd_check_matches_full_stiffness():
    rng = np.random.default_rng(0)
    targets = np.concatenate([rng.uniform(-6, -2, (500, 6)), rng.uniform(-0.9, 0.9, (500, 3))], axis=1)
    fast = is_positive_definite(targets)
    full = np.array([np.linalg.eigvalsh(stiffness_matrix(props_from_target(t)))[0] > 0 for t in targets])
    assert fast.any() and (~fast).any()
    assert (fast == full).all()


def test_report_fields(graphs):
    model = LatticeGNN(node_dim=1, edge_dim=graphs[0].edge_attr.size(1), global_dim=2, out_dim=9,
                       hidden=16, layers=1)
    model.fit_normalizers(graphs)
    result = evaluate(model, graphs, "val")
    for key in ("part", "n_lattices", "n_families", "loss", "overall", "targets", "pd",
                "worst_families", "families"):
        assert key in result
    assert result["n_lattices"] == len(graphs)
    assert result["pd"]["true_non_pd"] == 0
    exact = report(torch.cat([g.y for g in graphs]), torch.cat([g.y for g in graphs]),
                   [g.name for g in graphs], "val")
    assert exact["overall"]["median_percent_error_moduli"] == 0


def tiny(tmp_path, **extra):
    return {"paths": [SAMPLE], "epochs": 3, "hidden": 16, "layers": 1, "runs_dir": str(tmp_path),
            "verbose": False, **extra}


def test_train_writes_run_and_is_seeded(tmp_path, graphs):
    a = train(tiny(tmp_path, run_id="a"), graphs=graphs)
    b = train(tiny(tmp_path, run_id="b"), graphs=graphs)
    for name in ("config.json", "split.json", "model.pt", "history.json", "eval.json"):
        assert (tmp_path / a / name).exists()
    loss = [json.loads((tmp_path / r / "eval.json").read_text())["loss"] for r in (a, b)]
    assert loss[0] == pytest.approx(loss[1], rel=1e-5)


LOOP = {
    "data": {"paths": [SAMPLE], "split_file": None},
    "training": {"epochs": 2, "hidden": 16, "seed": 0},
    "search": {"lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
               "layers": {"type": "int", "low": 1, "high": 2}},
    "menu": {"augment": [True, False]},
    "optimizer": {"sampler_seed": 0, "objective": "loss"},
    "budget": {"runs": 3, "no_progress": 10},
    "thresholds": {"median_percent_error_moduli": 0.0},
}


def test_check_target():
    result = {"loss": 1.0, "overall": {"median_percent_error_moduli": 5.0}, "pd": {"non_pd_rate": 0.0}}
    budget = {"runs": 10, "no_progress": 3}
    assert check_target(result, {"median_percent_error_moduli": 10}, 1, budget)["reason"] == "target met"
    assert check_target(result, {"median_percent_error_moduli": 1}, 1, budget)["decision"] == "continue"
    assert "budget" in check_target(result, {"median_percent_error_moduli": 1}, 10, budget)["reason"]
    assert "progress" in check_target(result, {"median_percent_error_moduli": 1}, 5, budget, 3)["reason"]


def test_tiny_study(tmp_path, graphs):
    final = optimize(LOOP, "study", tmp_path, graphs=graphs)
    trials = json.loads((tmp_path / "study" / "trials.json").read_text())
    assert len(trials["trials"]) == 3
    assert final["check"]["reason"].startswith("budget")
    assert {"lr", "layers", "augment"} <= set(trials["trials"][0]["params"])
