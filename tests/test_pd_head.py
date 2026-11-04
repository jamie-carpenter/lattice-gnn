"""The pd_head switch: predicted couplings always give a positive definite stiffness."""
import json

import numpy as np
import pytest
import torch
from torch_geometric.loader import DataLoader

from lattice_gnn.data import load_dataset
from lattice_gnn.evaluate import is_positive_definite, normalized_block
from lattice_gnn.model import PD_MARGIN, LatticeGNN, pd_couplings
from lattice_gnn.training import train

SAMPLE = "data/samples/sample_lattices.txt"


@pytest.fixture(scope="module")
def graphs(tmp_path_factory):
    return load_dataset([SAMPLE], cache_dir=tmp_path_factory.mktemp("cache"))


def smallest_eigenvalue(couplings: torch.Tensor) -> np.ndarray:
    return np.linalg.eigvalsh(normalized_block(couplings.double().numpy()))[:, 0]


def targets_with(couplings: torch.Tensor) -> np.ndarray:
    """(n, 9) targets with arbitrary moduli: only the couplings decide positive definiteness."""
    return np.concatenate([np.zeros((len(couplings), 6)), couplings.double().numpy()], axis=1)


@pytest.mark.parametrize("scale", [0.1, 1.0, 10.0, 1e3, 1e8])
def test_pd_couplings_always_positive_definite(scale):
    a = torch.randn(20_000, 3, generator=torch.Generator().manual_seed(0)) * scale
    c = pd_couplings(a)
    assert is_positive_definite(targets_with(c)).all()
    assert smallest_eigenvalue(c).min() >= PD_MARGIN - 1e-6     # margin less float32 rounding


def test_pd_couplings_extreme_inputs():
    signs = torch.tensor([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=torch.float32)
    a = torch.cat([signs * 1e30, signs * float("inf"), torch.zeros(1, 3)])
    c = pd_couplings(a)
    assert torch.isfinite(c).all()
    assert is_positive_definite(targets_with(c)).all()


def test_pd_couplings_gradient_finite_when_saturated():
    a = torch.tensor([[50.0, -50.0, 50.0], [0.3, -2.0, 9.0]], requires_grad=True)
    pd_couplings(a).sum().backward()
    assert torch.isfinite(a.grad).all()


def test_pd_couplings_reach_every_positive_definite_block():
    """The head loses nothing: invert it on random PD blocks (smallest eigenvalue > margin)."""
    rng = np.random.default_rng(0)
    c = rng.uniform(-1, 1, (50_000, 3))
    c = c[smallest_eigenvalue(torch.as_tensor(c)) > 1e-3]
    R = -c / (1 - PD_MARGIN)                         # correlations [R12, R02, R01]
    r01, r02 = R[:, 2], R[:, 1]
    p12 = (R[:, 0] - r01 * r02) / np.sqrt((1 - r01 ** 2) * (1 - r02 ** 2))
    assert (np.abs(p12) < 1).all()
    a = np.arctanh(np.stack([p12, r02, r01], axis=1))
    back = pd_couplings(torch.as_tensor(a, dtype=torch.float64)).numpy()
    assert np.allclose(back, c, atol=1e-9)


def test_model_with_pd_head_never_non_pd(graphs):
    torch.manual_seed(0)
    first = graphs[0]
    model = LatticeGNN(first.x.size(1), first.edge_attr.size(1), first.u.size(1), 9, hidden=32,
                       layers=2, pd_head=True)
    model.fit_normalizers(graphs)
    with torch.no_grad():                            # push the coupling outputs far out
        model.head[-1].weight.mul_(1e3)
        model.head[-1].bias[6:].uniform_(-50, 50)
    model.eval()
    batch = next(iter(DataLoader(graphs, batch_size=len(graphs))))
    pred = model.predict(batch)
    assert is_positive_definite(pred.numpy()).all()
    with torch.no_grad():                            # forward is the standardized form of predict
        assert torch.allclose(model.y_norm.inverse(model(batch)), pred, atol=1e-4)


def test_pd_head_saved_and_loaded(tmp_path, graphs):
    first = graphs[0]
    dims = dict(node_dim=first.x.size(1), edge_dim=first.edge_attr.size(1), global_dim=first.u.size(1),
                out_dim=9, hidden=16, layers=1)
    LatticeGNN(**dims, pd_head=True).save(tmp_path / "m.pt")
    assert LatticeGNN.load(tmp_path / "m.pt")[0].pd_head
    old = LatticeGNN(**dims)                         # checkpoints from before the switch
    torch.save({"config": {k: v for k, v in old.config.items() if k != "pd_head"},
                "state_dict": old.state_dict()}, tmp_path / "old.pt")
    assert not LatticeGNN.load(tmp_path / "old.pt")[0].pd_head


def test_train_with_pd_head(tmp_path, graphs):
    run_id = train({"paths": [SAMPLE], "epochs": 3, "hidden": 16, "layers": 1, "pd_head": True,
                    "runs_dir": str(tmp_path), "verbose": False}, graphs=graphs)
    run = tmp_path / run_id
    assert json.loads((run / "config.json").read_text())["pd_head"] is True
    assert json.loads((run / "eval.json").read_text())["pd"]["non_pd"] == 0
