"""Check the diagnostician's decision in code before anything runs (agent_design_worksheet.md, sections 3 and 5).

The agent answers with one JSON object (DECISION_SCHEMA):
    decision     retune | change_structure | escalate
    reason       one or two sentences
    evidence     facts from the reports that support it (at least one)
    ranges       retune only: new search ranges, same format as `search` in config/loop.yaml
    menu_option  change_structure only: {"name": <menu switch>, "value": <one of its choices>}
    prediction   change_structure only: what should improve, and where
    would_try    escalate only: what it would try if allowed

validate(decision, state) returns a list of error messages (empty = valid) and whether code must
force an escalation instead of retrying. The loop (TODO section 10) gives the agent one retry
with the errors, then escalates. Rules:
  1. well-formed: matches DECISION_SCHEMA, and has the fields its decision type needs
  2. retune: each range is a setting in `limits`, of the right type, and inside its limits
  3. change_structure: the switch is on the menu, the value is one of its choices, it changes
     the current structure, and the cap on structural changes is not reached
  4. an option that already failed is not proposed again (forces escalation, no retry)
  5. retune and change_structure need a full round of runs left in the budget

`state` is built by state_from_loop(); escalate is always allowed if well-formed.

Usage:
    python -m lattice_gnn.validate --schema                    # print the schema (for claude -p)
    python -m lattice_gnn.validate decision.json state.json    # check one decision
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import jsonschema

DECISIONS = ("retune", "change_structure", "escalate")
NEEDS = {"retune": ("ranges",), "change_structure": ("menu_option", "prediction"),
         "escalate": ("would_try",)}

_RANGE = {
    "type": "object",
    "properties": {
        "type": {"enum": ["float", "int", "categorical"]},
        "low": {"type": "number"},
        "high": {"type": "number"},
        "log": {"type": "boolean"},
        "choices": {"type": "array", "minItems": 1, "items": {"type": ["number", "string", "boolean"]}},
    },
    "required": ["type"],
    "additionalProperties": False,
}

# Conditional fields (which decision needs which) are checked in code, not with if/then, so the
# same schema can be passed to `claude -p --json-schema` without relying on conditional support.
# No "$schema" key: Claude Code's checker rejects the draft 2020-12 URI. Only keywords that mean
# the same in draft-07 and 2020-12 are used, and validate() names its draft explicitly.
DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"enum": list(DECISIONS)},
        "reason": {"type": "string", "minLength": 1},
        "evidence": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
        "ranges": {"type": "object", "minProperties": 1, "additionalProperties": _RANGE},
        "menu_option": {
            "type": "object",
            "properties": {"name": {"type": "string"},
                           "value": {"type": ["number", "string", "boolean"]}},
            "required": ["name", "value"],
            "additionalProperties": False,
        },
        "prediction": {"type": "string", "minLength": 1},
        "would_try": {"type": "string", "minLength": 1},
    },
    "required": ["decision", "reason", "evidence"],
    "additionalProperties": False,
}


def state_from_loop(loop: dict, structure: dict | None = None, failed: list | None = None,
                    runs_used: int = 0, structural_changes: int = 0) -> dict:
    """What validate() needs, from config/loop.yaml plus where the loop is now.

    structure: current value of each menu switch (default: the first choice of each).
    failed: options already tried that did not help, as {"name", "value"} for menu options or
            {"name": "retune", "value": <ranges>} for retunes.
    """
    menu = loop.get("menu", {})
    return {
        "menu": menu,
        "limits": loop["limits"],
        "structure": structure if structure is not None else {k: v[0] for k, v in menu.items()},
        "failed": list(failed or []),
        "runs_used": runs_used,
        "budget": loop["budget"],
        "structural_changes": structural_changes,
    }


def _check_range(name: str, spec: dict, limits: dict) -> list[str]:
    if name not in limits:
        return [f"ranges.{name}: not a setting that can be retuned (allowed: {sorted(limits)})"]
    limit, kind = limits[name], spec["type"]
    if "choices" in limit:
        if kind != "categorical" or "choices" not in spec or {"low", "high", "log"} & set(spec):
            return [f"ranges.{name}: must be categorical with choices from {limit['choices']}"]
        outside = [c for c in spec["choices"] if c not in limit["choices"]]
        return [f"ranges.{name}: choices {outside} not allowed (allowed: {limit['choices']})"] if outside else []

    want = "int" if isinstance(limit["low"], int) and isinstance(limit["high"], int) else "float"
    if kind != want or "choices" in spec or "low" not in spec or "high" not in spec:
        return [f"ranges.{name}: must be {{type: {want}, low, high}}"]
    low, high, errors = spec["low"], spec["high"], []
    if not (math.isfinite(low) and math.isfinite(high)) or low >= high:
        errors.append(f"ranges.{name}: need low < high (got {low}, {high})")
    if low < limit["low"] or high > limit["high"]:
        errors.append(f"ranges.{name}: [{low}, {high}] outside the limits [{limit['low']}, {limit['high']}]")
    if want == "int" and not (float(low).is_integer() and float(high).is_integer()):
        errors.append(f"ranges.{name}: low and high must be whole numbers")
    if spec.get("log") and low <= 0:
        errors.append(f"ranges.{name}: a log range needs low > 0")
    return errors


def _same_option(a: dict, b: dict) -> bool:
    return a.get("name") == b.get("name") and a.get("value") == b.get("value")


def validate(decision, state: dict) -> dict:
    """{"ok": bool, "errors": [...], "force_escalate": bool} for one agent decision."""
    def result(errors, force=False):
        return {"ok": not errors, "errors": errors, "force_escalate": force}

    # 1. well-formed
    if not isinstance(decision, dict):
        return result(["decision must be a JSON object"])
    schema_errors = sorted(jsonschema.Draft202012Validator(DECISION_SCHEMA).iter_errors(decision),
                           key=lambda e: list(e.absolute_path))
    if schema_errors:
        return result([f"{'.'.join(map(str, e.absolute_path)) or '(top)'}: {e.message}" for e in schema_errors])
    kind = decision["decision"]
    errors = [f"{kind} needs '{f}'" for f in NEEDS[kind] if f not in decision]
    errors += [f"{kind} must not have '{f}'" for k, fields in NEEDS.items() if k != kind
               for f in fields if f in decision]
    if errors or kind == "escalate":
        return result(errors)

    # 4. already failed: forced escalation, no retry
    if kind == "change_structure":
        option = decision["menu_option"]
    else:
        option = {"name": "retune", "value": decision["ranges"]}
    if any(_same_option(option, f) for f in state["failed"]):
        return result([f"{option['name']} = {json.dumps(option['value'])} already failed"], force=True)

    # 2. retune ranges / 3. menu option
    if kind == "retune":
        for name, spec in decision["ranges"].items():
            errors += _check_range(name, spec, state["limits"])
    else:
        name, value = option["name"], option["value"]
        if name not in state["menu"]:
            errors.append(f"menu_option.name: {name!r} is not on the menu (menu: {sorted(state['menu'])})")
        elif value not in state["menu"][name] or type(value) is not type(state["menu"][name][0]):
            errors.append(f"menu_option.value: {value!r} is not a choice for {name} ({state['menu'][name]})")
        elif state["structure"].get(name) == value:
            errors.append(f"menu_option: {name} is already {value!r}; that changes nothing")
        cap = state["budget"].get("max_structural_changes", math.inf)
        if state["structural_changes"] >= cap:
            errors.append(f"cap of {cap} structural changes reached; escalate instead")

    # 5. budget
    left = state["budget"]["runs"] - state["runs_used"]
    need = state["budget"].get("round_runs", 1)
    if left < need:
        errors.append(f"budget: {left} runs left, a round needs {need}; escalate or stop instead")
    return result(errors)


def main():
    parser = argparse.ArgumentParser(description="Check one diagnostician decision.")
    parser.add_argument("decision", nargs="?", help="decision JSON file")
    parser.add_argument("state", nargs="?", help="state JSON file (see state_from_loop)")
    parser.add_argument("--schema", action="store_true", help="print DECISION_SCHEMA and exit")
    args = parser.parse_args()
    if args.schema:
        print(json.dumps(DECISION_SCHEMA, indent=1))
        return
    if not (args.decision and args.state):
        parser.error("give a decision file and a state file, or --schema")
    out = validate(json.loads(Path(args.decision).read_text()), json.loads(Path(args.state).read_text()))
    print(json.dumps(out, indent=1))
    raise SystemExit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
