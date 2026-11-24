"""Planted failures: label faults, the case configs, frozen repeats and the code-only scoring."""
import json

import pytest
import torch

from agents.diagnostician.agent import decide
from agents.diagnostician.planted import (case_loop, check_passes, frozen_calls, load_cases, matches,
                                          merge, score_folder, start_structure, train_round0, write_key)
from lattice_gnn.data import load_dataset
from lattice_gnn.training import plant_label_faults

SAMPLE = "data/samples/sample_lattices.txt"


@pytest.fixture(scope="module")
def graphs(tmp_path_factory):
    return load_dataset([SAMPLE], cache_dir=tmp_path_factory.mktemp("cache"))


# ---------------------------------------------------------------- label faults

def test_moduli_shift_copies_and_changes_only_that_family(graphs):
    before = [g.y.clone() for g in graphs]
    out = plant_label_faults(graphs, [{"family": "cub_Z06.0", "kind": "moduli_log10_shift", "value": 1.0}])
    for g, b, o in zip(graphs, before, out):
        assert torch.equal(g.y, b)                                 # the input is untouched
        if g.name.startswith("cub_Z06.0_"):
            assert torch.allclose(o.y[:, :6], b[:, :6] + 1.0) and torch.equal(o.y[:, 6:], b[:, 6:])
        else:
            assert o is g


def test_shuffle_permutes_labels_within_family(graphs):
    a = graphs[0].clone()
    b = graphs[1].clone()
    a.name, b.name = "fam_Z01.0_E1", "fam_Z01.0_E2"
    out = plant_label_faults([a, b], [{"family": "fam_Z01.0", "kind": "shuffle", "value": 1}])
    ys = sorted([tuple(o.y.flatten().tolist()) for o in out])
    assert ys == sorted([tuple(a.y.flatten().tolist()), tuple(b.y.flatten().tolist())])
    with pytest.raises(ValueError):
        plant_label_faults(graphs, [{"family": "nope", "kind": "shuffle", "value": 0}])


# ---------------------------------------------------------------- the case file

def test_cases_build_valid_loops():
    base, cases = load_cases()
    assert {c["kind"] for c in cases.values()} == {"known", "ambiguous"}
    for name, case in cases.items():
        loop = case_loop(base, case)
        assert loop["budget"]["cpu_hours"] == 22 and "planted" not in loop and "answer" not in loop
        for setting, spec in loop["search"].items():               # starting ranges inside the hard limits
            limit = loop["limits"][setting]
            if "choices" in limit:
                assert set(spec["choices"]) <= set(limit["choices"]), (name, setting)
            else:
                assert limit["low"] <= spec["low"] < spec["high"] <= limit["high"], (name, setting)
        assert set(start_structure(loop)) == set(loop["menu"])
    assert case_loop(base, cases["width_trap"])["budget"]["round_runs"] == 3
    assert case_loop(base, cases["width_trap"])["budget"]["max_structural_changes"] == 3   # merged, not replaced


def test_merge():
    assert merge({"a": {"x": 1, "y": 2}, "b": [1]}, {"a": {"y": 3}, "b": [2]}) == {"a": {"x": 1, "y": 3}, "b": [2]}


# ---------------------------------------------------------------- scoring

RETUNE_LOW = {"decision": "retune", "reason": "lr too high", "evidence": ["loss 0.9"],
              "ranges": {"lr": {"type": "float", "low": 1e-4, "high": 1e-3, "log": True},
                         "hidden": {"type": "categorical", "choices": [128, 256]}}}
PD = {"decision": "change_structure", "reason": "non-PD", "evidence": ["non-PD 0.1"],
      "menu_option": {"name": "pd_head", "value": True}, "prediction": "non-PD 0"}
ESC = {"decision": "escalate", "reason": "Family TET_Z04.8 errors near 900%", "evidence": ["x"],
       "would_try": "check its labels"}


def test_matches():
    assert matches(RETUNE_LOW, {"decision": "retune", "lr_high_max": 3e-3})
    assert not matches(RETUNE_LOW, {"decision": "retune", "lr_high_max": 5e-4})
    assert matches(RETUNE_LOW, {"lr_low_min": 1e-4}) and not matches(RETUNE_LOW, {"lr_low_min": 2e-4})
    assert matches(RETUNE_LOW, {"excludes": {"hidden": 512}})
    assert not matches(RETUNE_LOW, {"excludes": {"hidden": 256}})
    assert not matches(PD, {"excludes": {"hidden": 512}})          # no such range: not excluded
    assert matches(PD, {"menu_option": {"name": "pd_head", "value": True}})
    assert matches(ESC, {"decision": "escalate", "mentions": "tet_Z04.8"})   # case-insensitive
    assert not matches(ESC, {"mentions": "tet_Z04.0"})


def test_check_first_vs_any():
    first = {"when": "first", "match": {"decision": "retune"}}
    anywhere = {"when": "any", "match": {"decision": "retune"}}
    assert not check_passes(first, [PD, RETUNE_LOW]) and check_passes(anywhere, [PD, RETUNE_LOW])
    alt = {"when": "first", "any_of": [{"decision": "escalate"}, {"decision": "change_structure"}]}
    assert check_passes(alt, [PD]) and not check_passes(alt, [])


def test_score_folder_known_and_ambiguous(tmp_path):
    _, cases = load_cases()
    folder = tmp_path / "k"
    folder.mkdir()
    write_key(folder, "units_low_lr", cases["units_low_lr"], "loop")
    with open(folder / "decisions.jsonl", "w") as f:
        for d in (RETUNE_LOW, ESC):
            f.write(json.dumps({"decision": d, "forced": False}) + "\n")
    r = score_folder(folder)
    assert r["loop"]["checks"] == {"raises_lr": True, "names_family": True}

    folder = tmp_path / "a"
    folder.mkdir()
    write_key(folder, "control", cases["control"], "frozen")
    with open(folder / "repeats.jsonl", "w") as f:
        for i, d in enumerate((RETUNE_LOW, PD, ESC)):
            f.write(json.dumps({"decision": d, "forced": False, "decider": "agent", "repeat": i}) + "\n")
    r = score_folder(folder)
    assert [x["unsupported"] for x in r["repeats"]] == [False, True, False]


# ---------------------------------------------------------------- frozen repeats, end to end

def fake_agent(answer):
    def runner(command, prompt):
        runner.prompts.append(prompt)
        return json.dumps({"type": "result", "subtype": "success", "num_turns": 1, "duration_ms": 1,
                           "structured_output": answer, "usage": {}, "modelUsage": {}})
    runner.prompts = []
    return runner


def test_frozen_round_and_repeats(tmp_path, graphs):
    base, cases = load_cases()
    loop = case_loop(base, cases["units_low_lr"])
    loop["data"] = {"paths": [SAMPLE], "split_file": None}
    loop["training"] = {**loop["training"], "epochs": 1, "hidden": 16,
                        "label_faults": [{"family": "cub_Z06.0", "kind": "moduli_log10_shift", "value": 1.0}]}
    loop["search"] = {"lr": loop["search"]["lr"], "hidden": {"type": "categorical", "choices": [64]}}
    loop["budget"] = {**loop["budget"], "round_runs": 2}
    folder = tmp_path / "f"
    trials = train_round0(loop, folder, graphs)
    assert len(trials) == 2 and all(t["cpu_hours"] > 0 for t in trials)
    write_key(folder, "units_low_lr", cases["units_low_lr"], "frozen")

    runner = fake_agent(ESC)
    agent_fn = lambda f, lp, s: decide(f, lp, runner=runner, structure=s)  # noqa: E731
    entries = frozen_calls(folder, loop, start_structure(loop), 3, decide_fn=agent_fn, log=lambda *a: None)
    assert [e["decider"] for e in entries] == ["agent"] * 3 + ["rules"]
    assert not (folder / "decisions.jsonl").exists()
    # every call is a first decision: no earlier decisions in its log
    assert all("None yet: this is the first decision." in p for p in runner.prompts)
    # the answer key and the planted fault never reach the agent
    for p in runner.prompts:
        assert "label_faults" not in p and "moduli_log10_shift" not in p and "names_family" not in p
    r = score_folder(folder)
    agent = [x for x in r["repeats"] if x["decider"] == "agent"]
    assert all(x["checks"] == {"raises_lr": False, "names_family": True} for x in agent)
    assert len(r["repeats"]) == 4
