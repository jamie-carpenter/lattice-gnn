"""The agent loop end to end on the sample file, with a fake `claude` (no real agent calls)."""
import copy
import json
from pathlib import Path

import pytest
import yaml

from agents.diagnostician.agent import decide, read_jsonl
from agents.diagnostician.loop import run_loop
from lattice_gnn.data import load_dataset

SAMPLE = "data/samples/sample_lattices.txt"
REAL = yaml.safe_load(Path("config/loop.yaml").read_text())
LOOP = {**copy.deepcopy(REAL),                       # real menu, limits, agent and thresholds
        "data": {"paths": [SAMPLE], "split_file": None},
        "training": {"epochs": 1, "hidden": 16, "seed": 0},
        "search": {"lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
                   "layers": {"type": "int", "low": 1, "high": 2}},
        "budget": {"runs": 4, "round_runs": 2, "max_structural_changes": 3}}

RETUNE = {"decision": "retune", "reason": "lr at the edge", "evidence": ["best lr near 1e-2"],
          "ranges": {"lr": {"type": "float", "low": 2e-2, "high": 5e-2, "log": True}}}
PD = {"decision": "change_structure", "reason": "non-PD predictions", "evidence": ["non-PD > 0"],
      "menu_option": {"name": "pd_head", "value": True}, "prediction": "non-PD rate 0"}
ESCALATE = {"decision": "escalate", "reason": "labels look wrong", "evidence": ["one family"],
            "would_try": "check the labels"}


@pytest.fixture(scope="module")
def graphs(tmp_path_factory):
    return load_dataset([SAMPLE], cache_dir=tmp_path_factory.mktemp("cache"))


def agent_says(*answers):
    """decide_fn whose fake claude returns the given answers in order (and repeats the last)."""
    answers = list(answers)

    def runner(command, prompt):
        answer = answers.pop(0) if len(answers) > 1 else answers[0]
        return json.dumps({"type": "result", "subtype": "success", "num_turns": 1, "duration_ms": 10,
                           "structured_output": answer, "usage": {}, "modelUsage": {}})
    return lambda folder, loop, structure: decide(folder, loop, runner=runner, structure=structure)


def loop_run(tmp_path, graphs, decide_fn, loop=LOOP, **kw):
    summary = run_loop(copy.deepcopy(loop), "L", tmp_path, graphs=graphs, decide_fn=decide_fn,
                       log=lambda *a: None, **kw)
    folder = tmp_path / "L"
    return summary, json.loads((folder / "trials.json").read_text())["trials"], read_jsonl(folder / "decisions.jsonl")


def test_retune_applied_and_judged(tmp_path, graphs):
    summary, trials, decisions = loop_run(tmp_path, graphs, agent_says(RETUNE), approve="none")
    assert summary["stop"].startswith("budget") and summary["runs_used"] == 4 and summary["rounds"] == 2
    assert [t["round"] for t in trials] == [0, 0, 1, 1]
    assert all(2e-2 <= t["params"]["lr"] <= 5e-2 for t in trials[2:])          # new range used
    assert summary["search"]["lr"]["low"] == 2e-2
    (d,) = decisions
    assert d["approved"] and d["round"] == 0 and d["outcome"] in ("improved", "failed")
    assert set(d["outcome_metrics"]) >= {"value_before", "value_after", "non_pd_before", "non_pd_after"}
    for t in trials:
        assert (tmp_path / "L" / t["run_id"] / "eval.json").exists()


def test_structure_change_approved(tmp_path, graphs):
    summary, trials, decisions = loop_run(tmp_path, graphs, agent_says(PD), approve="all")
    assert summary["structure"]["pd_head"] is True
    assert [t["params"]["pd_head"] for t in trials] == [False, False, True, True]
    assert all(t["non_pd_rate"] == 0 for t in trials[2:])
    assert decisions[0]["approved"] and decisions[0]["outcome"] is not None


def test_structure_change_asked_and_refused(tmp_path, graphs):
    asked = []
    summary, trials, decisions = loop_run(tmp_path, graphs, agent_says(PD), approve="ask",
                                          ask=lambda q: asked.append(q) or "n")
    assert len(asked) == 1 and "pd_head" in asked[0]
    assert summary["structure"]["pd_head"] is False
    assert decisions[0]["approved"] is False and decisions[0]["outcome"] is None


def test_escalation_stops_unattended(tmp_path, graphs):
    summary, trials, decisions = loop_run(tmp_path, graphs, agent_says(ESCALATE), approve="all")
    assert summary["stop"].startswith("escalated") and summary["runs_used"] == 2


def test_escalation_continue_when_asked(tmp_path, graphs):
    summary, trials, _ = loop_run(tmp_path, graphs, agent_says(ESCALATE), approve="ask", ask=lambda q: "c")
    assert summary["stop"].startswith("budget") and summary["runs_used"] == 4


def test_failed_option_forces_escalation(tmp_path, graphs):
    loop = {**LOOP, "budget": {**LOOP["budget"], "runs": 6},
            "outcome": {"min_loss_drop": 1e9, "min_non_pd_drop": 1e9}}     # nothing counts as improved
    summary, trials, decisions = loop_run(tmp_path, graphs, agent_says(PD), loop=loop, approve="all")
    assert decisions[0]["outcome"] == "failed"
    assert decisions[1]["forced"] and "already failed" in decisions[1]["decision"]["reason"]
    assert summary["stop"].startswith("escalated") and summary["runs_used"] == 4
