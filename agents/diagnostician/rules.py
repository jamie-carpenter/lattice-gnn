"""Rule-table baseline: the diagnostician's decision made by a fixed script instead of an LLM.

Lab Report 2 review: the agent's first decision (pd_head when non-PD > 0) could be made by a
one-line rule, and the agent commits to one change per round while Optuna searches everything
at once. This decider separates the two: it runs inside the same loop (same rounds, budget,
validator, outcome check and decision log) and differs from the agent only in how it decides.

Rules, in order:
  1. Best run has non-PD predictions above the target, pd_head is off and has not failed
     -> change_structure: pd_head = true.
  2. Otherwise retune by zooming in on the best run so far (greedy sequential search):
       float, log scale: [best / 3, best * 3]
       float, linear:    [best - w/4, best + w/4], w = current width
       int:              [best - 1, best + 1]
       categorical:      the values of the 3 best runs; if the best value is the largest
                         (smallest) current choice, add the next larger (smaller) allowed value
     every range clipped to the hard limits in config/loop.yaml. If that exact zoom already
     failed, widen it (scale 2, 3, ...) instead of repeating it.
The validator checks the result like any agent answer; a rejected rule decision escalates.

Usage: python -m agents.diagnostician.loop --decider rules --name <name> --approve all
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from lattice_gnn.validate import validate

from .agent import load_target, read_jsonl, state_for

TOP_K = 3


def zoom(search: dict, limits: dict, best: dict, top: list[dict], scale: int = 1) -> dict:
    """New search ranges centred on the best run's settings, clipped to the hard limits.
    `scale` widens the ranges (log factor 3**scale, int +-scale, linear w*scale/4)."""
    ranges = {}
    for name, spec in search.items():
        if name not in best or name not in limits:
            continue
        value, limit = best[name], limits[name]
        if spec["type"] == "categorical":
            allowed = sorted(limit["choices"])
            current = sorted(spec["choices"])
            choices = {t[name] for t in top if name in t and t[name] in allowed}
            for edge, step in ((current[-1], 1), (current[0], -1)):
                i = allowed.index(value) if value in allowed else None
                if value == edge and i is not None and 0 <= i + step < len(allowed):
                    choices.add(allowed[i + step])
            ranges[name] = {"type": "categorical", "choices": sorted(choices)}
        elif spec["type"] == "int":
            ranges[name] = {"type": "int", "low": max(limit["low"], int(value) - scale),
                            "high": min(limit["high"], int(value) + scale)}
        elif spec.get("log"):
            ranges[name] = {"type": "float", "low": max(limit["low"], value / 3 ** scale),
                            "high": min(limit["high"], value * 3), "log": True}
        else:
            half = (spec["high"] - spec["low"]) * scale / 4
            ranges[name] = {"type": "float", "low": max(limit["low"], value - half),
                            "high": min(limit["high"], value + half)}
    return ranges


def rule_decision(target: dict, state: dict, loop: dict) -> dict:
    ev = target["eval"]
    non_pd = ev["pd"]["non_pd_rate"]
    goal = loop.get("thresholds", {}).get("non_pd_rate", 0.0)
    pd_failed = any(f.get("name") == "pd_head" for f in state["failed"])
    if non_pd > goal and "pd_head" in state["menu"] and not state["structure"].get("pd_head") \
            and not pd_failed:
        return {"decision": "change_structure",
                "reason": f"Rule 1: non-PD rate {non_pd:.4f} > target {goal} and pd_head is off.",
                "evidence": [f"best run non_pd_rate = {non_pd:.4f}"],
                "menu_option": {"name": "pd_head", "value": True},
                "prediction": "Rule table: no prediction."}

    trials = sorted(target["history"]["trials"], key=lambda t: t["value"])
    best = trials[0]["params"]
    failed = [f["value"] for f in state["failed"] if f.get("name") == "retune"]
    for scale in range(1, 6):   # a zoom that already failed is widened, not repeated
        ranges = zoom(loop["search"], loop["limits"], best,
                      [t["params"] for t in trials[:TOP_K * scale]], scale)
        if ranges not in failed:
            break
    return {"decision": "retune",
            "reason": f"Rule 2: zoom in on the best run so far ({trials[0]['run_id']}), width x{scale}.",
            "evidence": [f"best value {trials[0]['value']:.4f}; top {TOP_K} runs: "
                         + ", ".join(t["run_id"] for t in trials[:TOP_K])],
            "ranges": ranges}


def decide_rules(folder: str | Path, loop: dict, structure: dict | None = None) -> dict:
    """Same contract as agent.decide: append one entry to decisions.jsonl and return it."""
    folder = Path(folder)
    decisions = read_jsonl(folder / "decisions.jsonl")
    target = load_target(folder)
    state = state_for(target, loop, decisions, structure)
    answer = rule_decision(target, state, loop)
    check = validate(answer, state)
    decision = answer if check["ok"] else {
        "decision": "escalate", "reason": "Forced by code: the rule decision failed validation.",
        "evidence": check["errors"], "would_try": "No rule applies."}
    entry = {"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "decision": decision, "forced": not check["ok"], "decider": "rules",
             "attempts": [{"call": 0, "answer": answer, "errors": check["errors"]}],
             "approved": None, "outcome": None}
    with open(folder / "decisions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry
