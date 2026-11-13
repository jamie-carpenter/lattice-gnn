"""validate.py: one test (or more) per rule, on the real config/loop.yaml."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from lattice_gnn.validate import DECISION_SCHEMA, state_from_loop, validate

LOOP = yaml.safe_load(Path("config/loop.yaml").read_text())

RETUNE = {"decision": "retune", "reason": "best lr sits at the upper edge",
          "evidence": ["best trial lr 2.9e-3, range high 3e-3"],
          "ranges": {"lr": {"type": "float", "low": 1e-3, "high": 1e-2, "log": True}}}
CHANGE = {"decision": "change_structure", "reason": "non-PD predictions persist after retuning",
          "evidence": ["non_pd_rate 0.095 in every trial"],
          "menu_option": {"name": "pd_head", "value": True},
          "prediction": "non-PD rate drops to 0; coupling errors within seed spread"}
ESCALATE = {"decision": "escalate", "reason": "one family's errors look like bad labels",
            "evidence": ["mon_Z05.1: 181% median error, others < 60%"],
            "would_try": "check that family's labels against the catalog"}


@pytest.fixture
def state():
    return state_from_loop(LOOP, structure={"augment": True, "pd_head": False}, runs_used=5)


def check(decision, state):
    return validate(copy.deepcopy(decision), state)


@pytest.mark.parametrize("decision", [RETUNE, CHANGE, ESCALATE])
def test_valid_decisions_pass(decision, state):
    assert check(decision, state) == {"ok": True, "errors": [], "force_escalate": False}


# Rule 1: well-formed
@pytest.mark.parametrize("bad", [
    "retune",                                                  # not an object
    {"decision": "retune"},                                    # no reason / evidence
    {**ESCALATE, "decision": "retrain"},                       # unknown decision
    {**ESCALATE, "evidence": []},                              # no evidence
    {**ESCALATE, "confidence": 0.9},                           # unknown field
    {**RETUNE, "ranges": {}},                                  # empty ranges
    {**RETUNE, "ranges": {"lr": {"type": "float", "low": 1e-3, "high": 1e-2, "step": 1}}},
    {**CHANGE, "menu_option": {"name": "pd_head"}},            # no value
])
def test_rule1_schema(bad, state):
    out = check(bad, state)
    assert not out["ok"] and not out["force_escalate"]


@pytest.mark.parametrize("bad, message", [
    ({k: v for k, v in RETUNE.items() if k != "ranges"}, "retune needs 'ranges'"),
    ({k: v for k, v in CHANGE.items() if k != "prediction"}, "change_structure needs 'prediction'"),
    ({k: v for k, v in ESCALATE.items() if k != "would_try"}, "escalate needs 'would_try'"),
    ({**RETUNE, "menu_option": CHANGE["menu_option"]}, "retune must not have 'menu_option'"),
])
def test_rule1_fields_per_decision(bad, message, state):
    assert message in check(bad, state)["errors"]


# Rule 2: retune ranges
@pytest.mark.parametrize("ranges, fragment", [
    ({"dropout": {"type": "float", "low": 0.0, "high": 0.5}}, "not a setting"),
    ({"lr": {"type": "float", "low": 1e-6, "high": 1e-3, "log": True}}, "outside the limits"),
    ({"lr": {"type": "float", "low": 1e-3, "high": 1.0}}, "outside the limits"),
    ({"lr": {"type": "float", "low": 1e-2, "high": 1e-3}}, "low < high"),
    ({"lr": {"type": "categorical", "choices": [1e-3]}}, "must be {type: float"),
    ({"layers": {"type": "int", "low": 2, "high": 12}}, "outside the limits"),
    ({"layers": {"type": "float", "low": 2, "high": 6}}, "must be {type: int"),
    ({"layers": {"type": "int", "low": 2.5, "high": 6}}, "whole numbers"),
    ({"weight_decay": {"type": "float", "low": 0.0, "high": 1e-3, "log": True}}, "outside the limits"),
    ({"hidden": {"type": "categorical", "choices": [128, 1024]}}, "[1024] not allowed"),
    ({"batch_size": {"type": "int", "low": 16, "high": 64}}, "must be categorical"),
])
def test_rule2_ranges(ranges, fragment, state):
    out = check({**RETUNE, "ranges": ranges}, state)
    assert not out["ok"] and any(fragment in e for e in out["errors"]), out["errors"]


def test_rule2_limits_contain_search():
    """The starting search ranges must themselves pass, or a 'keep these' retune would fail."""
    out = check({**RETUNE, "ranges": LOOP["search"]}, state_from_loop(LOOP))
    assert out["ok"], out["errors"]


# Rule 3: menu option
@pytest.mark.parametrize("option, fragment", [
    ({"name": "dropout", "value": True}, "not on the menu"),
    ({"name": "pd_head", "value": "yes"}, "not a choice"),
    ({"name": "pd_head", "value": 1}, "not a choice"),          # 1 == True, but not a bool
    ({"name": "augment", "value": True}, "already True"),
])
def test_rule3_menu_option(option, fragment, state):
    out = check({**CHANGE, "menu_option": option}, state)
    assert not out["ok"] and any(fragment in e for e in out["errors"]), out["errors"]


def test_rule3_structural_cap(state):
    state["structural_changes"] = LOOP["budget"]["max_structural_changes"]
    out = check(CHANGE, state)
    assert any("cap of" in e for e in out["errors"])
    assert check(ESCALATE, state)["ok"]


# Rule 4: already failed -> forced escalation
def test_rule4_failed_menu_option(state):
    state["failed"].append({"name": "pd_head", "value": True})
    out = check(CHANGE, state)
    assert not out["ok"] and out["force_escalate"]


def test_rule4_failed_retune(state):
    state["failed"].append({"name": "retune", "value": copy.deepcopy(RETUNE["ranges"])})
    assert check(RETUNE, state)["force_escalate"]
    other = {**RETUNE, "ranges": {"lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True}}}
    assert check(other, state)["ok"]


# Rule 5: budget
@pytest.mark.parametrize("decision", [RETUNE, CHANGE])
def test_rule5_budget(decision, state):
    state["runs_used"] = LOOP["budget"]["runs"] - LOOP["budget"]["round_runs"] + 1
    out = check(decision, state)
    assert any("budget" in e for e in out["errors"])
    assert check(ESCALATE, state)["ok"]


def test_schema_and_cli(tmp_path, state):
    out = subprocess.run([sys.executable, "-m", "lattice_gnn.validate", "--schema"],
                         capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == DECISION_SCHEMA
    (tmp_path / "d.json").write_text(json.dumps(CHANGE))
    (tmp_path / "s.json").write_text(json.dumps(state))
    ok = subprocess.run([sys.executable, "-m", "lattice_gnn.validate", str(tmp_path / "d.json"),
                         str(tmp_path / "s.json")], capture_output=True, text=True)
    assert ok.returncode == 0 and json.loads(ok.stdout)["ok"]
