"""Run the diagnostician once: build its input, call `claude -p`, validate, log.

The agent (.claude/agents/diagnostician.md) has no tools: code gives it every number it may use
and checks its answer with lattice_gnn.validate before anything runs. Model, effort level, turn
limit and retries come from the `agent` section of config/loop.yaml.

Run from the repo root on a training run (runs/<id>/) or an Optuna study (runs/<study>/):
    python -m agents.diagnostician runs/main-full
    python -m agents.diagnostician runs/main-full --prompt-only     # write the input, no call

Writes, in that folder:
  agent/prompt-<n>.md      the exact input of each call (n counts calls in this folder)
  agent/response-<n>.json  Claude Code's raw JSON result
  decisions.jsonl          one line per decision: the decision, validation, whether code forced it
  run_info.json            per call: model requested and used, Claude Code version, effort,
                           tokens, cost estimate, duration
The loop (TODO section 10) fills in `approved` and `outcome` in decisions.jsonl later.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from lattice_gnn.training import DEFAULTS
from lattice_gnn.validate import DECISION_SCHEMA, state_from_loop, validate

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- reading the loop's files

def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def load_target(folder: Path) -> dict:
    """Best run's config and eval.json, plus the trial history if `folder` is a study."""
    trials_file = folder / "trials.json"
    if trials_file.exists():
        trials = json.loads(trials_file.read_text())
        run = folder / trials["best"]["run_id"]
        history = trials
        runs_used = len(trials["trials"])
    else:
        run, history, runs_used = folder, None, 1
    return {"config": json.loads((run / "config.json").read_text()),
            "eval": json.loads((run / "eval.json").read_text()),
            "history": history, "runs_used": runs_used}


def state_for(target: dict, loop: dict, decisions: list[dict], structure: dict | None = None) -> dict:
    """validate()'s state: history from the decision log; structure given by the loop, else from
    the best run's config (a single run, or before any structural change)."""
    if structure is None:
        config = {**DEFAULTS, **target["config"]}
        structure = {name: config[name] for name in loop.get("menu", {})}
    applied = [d for d in decisions if d.get("approved") and not d.get("forced")]
    failed = []
    for d in decisions:
        if d.get("outcome") != "failed":
            continue
        dec = d["decision"]
        if dec["decision"] == "change_structure":
            failed.append(dict(dec["menu_option"]))
        elif dec["decision"] == "retune":
            failed.append({"name": "retune", "value": dec["ranges"]})
    return state_from_loop(loop, structure=structure, failed=failed, runs_used=target["runs_used"],
                           structural_changes=sum(d["decision"]["decision"] == "change_structure" for d in applied))


# ---------------------------------------------------------------- the agent's input

def _block(title: str, value) -> str:
    return f"## {title}\n\n```json\n{json.dumps(value, indent=1)}\n```\n"


def build_prompt(target: dict, state: dict, loop: dict, decisions: list[dict]) -> str:
    """One message with everything the agent may use (agent_design_worksheet.md section 3)."""
    budget = state["budget"]
    state_view = {
        "structure": state["structure"], "menu": state["menu"],
        "search_ranges": loop["search"], "hard_limits": state["limits"],
        "budget": {"runs_total": budget["runs"], "runs_used": state["runs_used"],
                   "runs_per_round": budget.get("round_runs"),
                   "structural_changes_used": state["structural_changes"],
                   "structural_changes_cap": budget.get("max_structural_changes")},
        "failed_options": state["failed"],
        "targets": {"thresholds": loop["thresholds"], "optimizer_objective": loop["optimizer"]["objective"]},
    }
    config = {k: target["config"].get(k) for k in
              ("epochs", "patience", "lr", "weight_decay", "hidden", "layers", "batch_size", "seed")}
    report = {**target["eval"], "best_run_settings": config}
    if target["history"] is None:
        history = ("None yet: this is a single training run with the default settings above, "
                   "not an optimizer round.")
    else:
        history = {"objective": target["history"]["objective"], "best": target["history"]["best"],
                   "trials": [{k: t[k] for k in ("round", "number", "params", "value", "best_epoch",
                                                 "overall", "non_pd_rate") if k in t}
                              for t in target["history"]["trials"]]}
    log = [{k: d.get(k) for k in ("decision", "forced", "approved", "outcome")} for d in decisions] \
        or "None yet: this is the first decision."
    return "\n".join([
        "Decide the next step for the surrogate. Answer with one JSON object, as your instructions say.\n",
        _block("1. State", state_view),
        _block("2. Evaluation report (validation families, best run)", report),
        _block("3. Optimizer history", history),
        _block("4. Decision log", log),
    ])


def retry_prompt(prompt: str, answer, errors: list[str]) -> str:
    return (prompt + "\n## Your previous answer was rejected\n\n"
            f"```json\n{json.dumps(answer, indent=1)}\n```\n\nErrors:\n"
            + "\n".join(f"- {e}" for e in errors) + "\n\nAnswer again, fixing these errors.\n")


# ---------------------------------------------------------------- calling Claude Code

def build_command(claude: str, agent: dict) -> list[str]:
    return [
        claude, "-p",                                   # prompt on stdin (too long for a Windows command line)
        "--agent", agent["name"],                       # instructions from .claude/agents/
        "--model", agent["model"], "--effort", agent["effort"],
        "--max-turns", str(agent["max_turns"]),
        "--json-schema", json.dumps(DECISION_SCHEMA),
        "--output-format", "json",
        "--tools", "",                                  # no tools at all
        "--mcp-config", json.dumps({"mcpServers": {}}), "--strict-mcp-config",
        "--permission-mode", "dontAsk",
        "--no-session-persistence",
    ]


def run_claude(command: list[str], prompt: str) -> str:
    proc = subprocess.run(command, input=prompt, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", cwd=REPO_ROOT)
    if proc.returncode != 0 and not proc.stdout.strip():
        raise RuntimeError(f"claude exited with code {proc.returncode}: {proc.stderr.strip()[:500]}")
    return proc.stdout


def parse_result(stdout: str) -> tuple[object, dict]:
    """(the agent's answer, call info) from `claude -p --output-format json`."""
    result = json.loads(stdout)
    answer = result.get("structured_output")
    if answer is None:                                  # fall back to the text answer
        text = (result.get("result") or "").strip().removeprefix("```json").removesuffix("```").strip()
        try:
            answer = json.loads(text)
        except json.JSONDecodeError:
            answer = text
    usage = result.get("usage") or {}
    info = {
        "is_error": bool(result.get("is_error")), "subtype": result.get("subtype"),
        "turns": result.get("num_turns"), "duration_s": round((result.get("duration_ms") or 0) / 1000, 1),
        "models_used": sorted((result.get("modelUsage") or {}).keys()),
        "tokens": {k: usage.get(k) for k in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                                             "cache_creation_input_tokens")},
        # Client-side estimate; usage is covered by the Claude Code seat, not billed per token.
        "cost_estimate_usd": result.get("total_cost_usd"),
        "session_id": result.get("session_id"),
    }
    return answer, info


# ---------------------------------------------------------------- one decision

def decide(folder: str | Path, loop: dict, claude: str = "claude", runner=run_claude,
           prompt_only: bool = False, structure: dict | None = None) -> dict | None:
    """Ask for one decision on `folder`, with up to agent.retries retries; returns the logged entry.

    `loop["search"]` is shown as the current search ranges; the loop passes its retuned ranges
    there, and the current structure in `structure`.
    """
    folder = Path(folder)
    agent = loop["agent"]
    decisions = read_jsonl(folder / "decisions.jsonl")
    target = load_target(folder)
    state = state_for(target, loop, decisions, structure)
    prompt = build_prompt(target, state, loop, decisions)
    out = folder / "agent"
    out.mkdir(exist_ok=True)
    n = len(list(out.glob("prompt-*.md")))
    if prompt_only:
        (out / f"prompt-{n}.md").write_text(prompt, encoding="utf-8")
        return None

    info_file = folder / "run_info.json"
    run_info = json.loads(info_file.read_text()) if info_file.exists() else {"calls": []}
    version = subprocess.run([claude, "--version"], capture_output=True, text=True).stdout.strip() \
        if runner is run_claude else "test"
    command = build_command(claude, agent)

    attempts, answer, check = [], None, None
    for attempt in range(agent.get("retries", 1) + 1):
        if attempt:
            prompt = retry_prompt(prompt, answer, check["errors"])
        (out / f"prompt-{n}.md").write_text(prompt, encoding="utf-8")
        stdout = runner(command, prompt)
        (out / f"response-{n}.json").write_text(stdout, encoding="utf-8")
        answer, info = parse_result(stdout)
        check = validate(answer, state)
        run_info["calls"].append({
            "call": n, "attempt": attempt,
            "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_requested": agent["model"], "effort": agent["effort"], "claude_code_version": version,
            **info, "valid": check["ok"],
        })
        info_file.write_text(json.dumps(run_info, indent=1))
        attempts.append({"call": n, "answer": answer, "errors": check["errors"]})
        n += 1
        if check["ok"] or check["force_escalate"]:
            break

    forced = not check["ok"]
    decision = answer if check["ok"] else {
        "decision": "escalate",
        "reason": "Forced by code: " + ("the agent proposed an option that already failed."
                                        if check["force_escalate"] else
                                        f"the agent's output failed validation {len(attempts)} times."),
        "evidence": check["errors"], "would_try": "See the rejected answers in `attempts`."}
    entry = {"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "decision": decision, "forced": forced, "attempts": attempts,
             "approved": None, "outcome": None}
    with open(folder / "decisions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def main():
    parser = argparse.ArgumentParser(description="Ask the diagnostician for one decision.")
    parser.add_argument("folder", help="training run or study folder, e.g. runs/main-full")
    parser.add_argument("--config", default="config/loop.yaml")
    parser.add_argument("--prompt-only", action="store_true", help="write the agent's input; no call")
    args = parser.parse_args()

    loop = yaml.safe_load(Path(args.config).read_text())
    claude = shutil.which("claude")
    if claude is None and not args.prompt_only:
        sys.exit("Claude Code (`claude`) not found on PATH.")
    entry = decide(args.folder, loop, claude or "claude", prompt_only=args.prompt_only)
    if entry is None:
        print(f"Wrote the agent's input to {Path(args.folder) / 'agent'}")
        return
    print(json.dumps(entry["decision"], indent=1))
    print(f"\n{'FORCED ' if entry['forced'] else ''}{entry['decision']['decision']}, after "
          f"{len(entry['attempts'])} call(s). Logged in {Path(args.folder) / 'decisions.jsonl'}")
