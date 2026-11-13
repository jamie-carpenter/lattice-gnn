"""The diagnostician's plumbing with a fake `claude`: input, retries, forced escalation, logs.

The real agent call is not tested here (it needs Claude Code and a sign-in).
"""
import json
from pathlib import Path

import pytest
import yaml

from agents.diagnostician.agent import build_command, decide, load_target, read_jsonl, state_for

LOOP = yaml.safe_load(Path("config/loop.yaml").read_text())
EVAL = {"part": "val", "n_lattices": 3, "n_families": 1, "loss": 0.11, "best_epoch": 97,
        "overall": {"median_percent_error_moduli": 30.8, "median_abs_error_couplings": 0.046},
        "targets": {}, "pd": {"threshold": 3.6e-7, "non_pd": 1, "non_pd_rate": 0.33, "true_non_pd": 0},
        "worst_families": ["tet_Z04.8"],
        "families": {"tet_Z04.8": {"n": 3, "median_percent_error_moduli": 30.8,
                                   "median_abs_error_couplings": 0.05, "non_pd": 1}}}
CHANGE = {"decision": "change_structure", "reason": "non-PD predictions",
          "evidence": ["non_pd_rate 0.33"], "menu_option": {"name": "pd_head", "value": True},
          "prediction": "non-PD rate 0 on tet_Z04.8"}


@pytest.fixture
def run(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"lr": 1e-3, "augment": True, "epochs": 100}))
    (tmp_path / "eval.json").write_text(json.dumps(EVAL))
    return tmp_path


def fake_claude(*answers, cost=0.01):
    """A runner that returns the given answers in order, in claude -p's JSON format."""
    answers, prompts = list(answers), []

    def runner(command, prompt):
        prompts.append(prompt)
        return json.dumps({"type": "result", "subtype": "success", "is_error": False, "num_turns": 1,
                           "duration_ms": 1500, "structured_output": answers.pop(0),
                           "usage": {"input_tokens": 5000, "output_tokens": 300},
                           "modelUsage": {LOOP["agent"]["model"]: {}}, "total_cost_usd": cost})
    runner.prompts = prompts
    return runner


def test_command_pins_model_effort_and_no_tools():
    cmd = build_command("claude", LOOP["agent"])
    assert cmd[cmd.index("--model") + 1] == LOOP["agent"]["model"] == "claude-sonnet-5-5"
    assert cmd[cmd.index("--effort") + 1] == LOOP["agent"]["effort"] == "high"
    assert cmd[cmd.index("--tools") + 1] == ""
    assert "--json-schema" in cmd and "--strict-mcp-config" in cmd


def test_state_from_run_defaults_missing_switches(run):
    state = state_for(load_target(run), LOOP, [])
    assert state["structure"] == {"augment": True, "pd_head": False}   # pd_head absent -> default
    assert state["runs_used"] == 1 and state["failed"] == [] and state["structural_changes"] == 0


def test_valid_decision_logged(run):
    runner = fake_claude(CHANGE)
    entry = decide(run, LOOP, runner=runner)
    assert entry["decision"] == CHANGE and not entry["forced"] and len(entry["attempts"]) == 1
    prompt = runner.prompts[0]
    assert '"non_pd_rate": 0.33' in prompt and "None yet: this is the first decision." in prompt
    assert read_jsonl(run / "decisions.jsonl") == [entry]
    call = json.loads((run / "run_info.json").read_text())["calls"][0]
    assert call["valid"] and call["effort"] == "high" and call["tokens"]["input_tokens"] == 5000
    assert (run / "agent" / "prompt-0.md").exists() and (run / "agent" / "response-0.json").exists()


def test_retry_then_valid(run):
    bad = {**CHANGE, "menu_option": {"name": "dropout", "value": True}}
    runner = fake_claude(bad, CHANGE)
    entry = decide(run, LOOP, runner=runner)
    assert entry["decision"] == CHANGE and not entry["forced"] and len(entry["attempts"]) == 2
    assert "not on the menu" in runner.prompts[1] and '"dropout"' in runner.prompts[1]
    assert len(json.loads((run / "run_info.json").read_text())["calls"]) == 2


def test_invalid_twice_forces_escalation(run):
    entry = decide(run, LOOP, runner=fake_claude("not json", {"decision": "retrain"}))
    assert entry["forced"] and entry["decision"]["decision"] == "escalate"
    assert "failed validation 2 times" in entry["decision"]["reason"]


def test_failed_option_forces_escalation_without_retry(run):
    (run / "decisions.jsonl").write_text(json.dumps(
        {"decision": CHANGE, "forced": False, "approved": True, "outcome": "failed"}) + "\n")
    runner = fake_claude(CHANGE, CHANGE)
    entry = decide(run, LOOP, runner=runner)
    assert entry["forced"] and len(runner.prompts) == 1 and "already failed" in entry["decision"]["reason"]
    assert '"outcome": "failed"' in runner.prompts[0]          # the agent sees its own history
    assert len(read_jsonl(run / "decisions.jsonl")) == 2


def test_prompt_only_makes_no_call(run):
    assert decide(run, LOOP, runner=fake_claude(), prompt_only=True) is None
    assert (run / "agent" / "prompt-0.md").exists() and not (run / "decisions.jsonl").exists()
