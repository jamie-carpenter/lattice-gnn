"""The rule-table baseline (agents/diagnostician/rules.py): each rule, and one loop end to end."""
import copy

from agents.diagnostician.rules import decide_rules, rule_decision, zoom
from lattice_gnn.validate import state_from_loop, validate
from test_loop import LOOP, graphs, loop_run  # noqa: F401  (graphs is a fixture)

SEARCH = {"lr": {"type": "float", "low": 1e-4, "high": 3e-3, "log": True},
          "layers": {"type": "int", "low": 2, "high": 6},
          "hidden": {"type": "categorical", "choices": [64, 128, 256]}}
LIMITS = {"lr": {"low": 1e-5, "high": 1e-1}, "layers": {"low": 1, "high": 8},
          "hidden": {"choices": [32, 64, 128, 256]}}
CFG = {**copy.deepcopy(LOOP), "search": SEARCH, "limits": LIMITS,
       "budget": {"runs": 20, "round_runs": 5, "max_structural_changes": 3}}


def target(non_pd, trials):
    return {"eval": {"pd": {"non_pd_rate": non_pd}}, "history": {"trials": trials}}


def trial(n, value, **params):
    return {"run_id": f"trial-{n:03d}", "value": value, "params": params}


TRIALS = [trial(0, 0.3, lr=1e-3, layers=4, hidden=128), trial(1, 0.2, lr=5e-4, layers=6, hidden=256),
          trial(2, 0.25, lr=2e-3, layers=3, hidden=64), trial(3, 0.4, lr=1e-4, layers=2, hidden=64)]


def state(**kw):
    return state_from_loop(CFG, **kw)


def test_rule1_pd_head_when_non_pd():
    d = rule_decision(target(0.05, TRIALS), state(structure={"augment": True, "pd_head": False}), CFG)
    assert d["decision"] == "change_structure" and d["menu_option"] == {"name": "pd_head", "value": True}


def test_rule2_retune_when_pd_head_on_or_failed():
    for st in (state(structure={"augment": True, "pd_head": True}),
               state(structure={"augment": True, "pd_head": False},
                     failed=[{"name": "pd_head", "value": True}])):
        assert rule_decision(target(0.05, TRIALS), st, CFG)["decision"] == "retune"


def test_zoom_centres_on_best_and_stays_in_limits():
    d = rule_decision(target(0.0, TRIALS), state(structure={"augment": True, "pd_head": True}), CFG)
    r = d["ranges"]
    assert r["lr"]["low"] < 5e-4 < r["lr"]["high"] and abs(r["lr"]["high"] / r["lr"]["low"] - 9) < 1e-9
    assert r["layers"] == {"type": "int", "low": 5, "high": 7}
    assert r["hidden"]["choices"] == [64, 128, 256]       # top 3; 256 is the cap, nothing added
    assert validate(d, state(structure={"augment": True, "pd_head": True}))["ok"]


def test_zoom_adds_next_choice_at_the_edge_and_clips():
    r = zoom(SEARCH, LIMITS, {"lr": 9e-2, "layers": 1, "hidden": 64}, [{"hidden": 64}])
    assert r["hidden"]["choices"] == [32, 64]
    assert r["lr"]["high"] == 1e-1 and r["layers"] == {"type": "int", "low": 1, "high": 2}


def test_failed_zoom_is_widened_not_repeated():
    st = state(structure={"augment": True, "pd_head": True})
    first = rule_decision(target(0.0, TRIALS), st, CFG)["ranges"]
    st["failed"] = [{"name": "retune", "value": first}]
    second = rule_decision(target(0.0, TRIALS), st, CFG)
    assert second["ranges"] != first and second["ranges"]["layers"] == {"type": "int", "low": 4, "high": 8}
    assert validate(second, st)["ok"]


def test_rules_loop_end_to_end(tmp_path, graphs):  # noqa: F811
    decide = lambda folder, loop, structure: decide_rules(folder, loop, structure)  # noqa: E731
    summary, trials, decisions = loop_run(tmp_path, graphs, decide, approve="all")
    assert summary["runs_used"] == 4 and summary["stop"].startswith("budget")
    (d,) = decisions
    assert d["decider"] == "rules" and not d["forced"] and d["approved"]
    assert d["outcome"] in ("improved", "failed")
