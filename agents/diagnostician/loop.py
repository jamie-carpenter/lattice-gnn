"""The agent loop (agent_design_worksheet.md section 6): optimize -> evaluate -> check -> agent ->
validate -> human gate -> apply -> repeat. Plain Python; all state lives in JSON files.

Each round, Optuna runs `budget.round_runs` training runs over the current search ranges, with
the structure (menu switches) held fixed. Unlike the baseline (lattice_gnn.optimize), the
optimizer never touches the menu here: only the agent's change_structure decisions do.
Each round is a new Optuna study (sampler seed = optimizer.sampler_seed + round), since a retune
changes the search space; at 5 runs per round TPE is still in its random start-up phase anyway.

Human gate (--approve):
  ask   change_structure needs a yes at the terminal; an escalation asks continue or stop
  all   unattended: every change_structure is approved; an escalation stops the loop
  none  unattended: every change_structure is refused; an escalation stops the loop
Retunes run without approval (worksheet section 7). An escalation counts as "not solved".

After the round that follows a decision, code fills in its outcome (worksheet section 3,
check 3): "improved" if the round's best run beats the best run before the decision by more
than `outcome.min_loss_drop` in the objective, or lowers the non-PD rate by more than
`outcome.min_non_pd_drop`; otherwise "failed". A failed option cannot be proposed again.

Usage (repo root):
    python -m agents.diagnostician.loop --config config/loop.yaml --name agent-0
    python -m agents.diagnostician.loop --name tiny --runs 4 --epochs 2 --approve all

Writes runs/<name>/: loop.json (settings), round-<r>/trial-<n>/ (one folder per training run),
trials.json (all runs, the agent's optimizer history), decisions.jsonl and run_info.json (from
the agent), agent/ (its prompts and replies), summary.json (why it stopped, best run, final
structure and ranges).
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import optuna
import yaml

from lattice_gnn.data import load_dataset
from lattice_gnn.optimize import check_target, score, suggest
from lattice_gnn.training import git_commit, train

from .agent import decide, read_jsonl

APPROVE_MODES = ("ask", "all", "none")


def run_round(loop: dict, search: dict, structure: dict, folder: Path, round_no: int,
              first_number: int, n_runs: int, graphs) -> list[dict]:
    """n_runs training runs over `search` with the structure fixed; one trials.json entry each."""
    objective = loop["optimizer"]["objective"]
    sampler = optuna.samplers.TPESampler(seed=loop["optimizer"]["sampler_seed"] + round_no)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    round_dir = folder / f"round-{round_no}"
    trials = []

    def objective_fn(trial: optuna.Trial) -> float:
        params = suggest(trial, search, {})
        number = first_number + trial.number
        config = {**loop["training"], **structure, **params, "paths": loop["data"]["paths"],
                  "split_file": loop["data"]["split_file"], "runs_dir": str(round_dir),
                  "run_id": f"trial-{number:03d}", "verbose": False}
        train(config, graphs=graphs)
        result = json.loads((round_dir / f"trial-{number:03d}" / "eval.json").read_text())
        value = score(result, objective)
        trials.append({"round": round_no, "number": number, "run_id": f"round-{round_no}/trial-{number:03d}",
                       "params": {**params, **structure}, "value": value,
                       "best_epoch": result["best_epoch"], "overall": result["overall"],
                       "non_pd_rate": result["pd"]["non_pd_rate"]})
        return value

    study.optimize(objective_fn, n_trials=n_runs)
    return trials


def gate(decision: dict, mode: str, ask) -> bool:
    """Human gate. True: apply the change_structure / continue after the escalation."""
    kind = decision["decision"]
    if kind == "retune":
        return True
    if mode != "ask":
        return kind == "change_structure" and mode == "all"
    print("\n" + json.dumps(decision, indent=1))
    if kind == "change_structure":
        option = decision["menu_option"]
        return ask(f"Approve {option['name']} = {json.dumps(option['value'])}? [y/N] ").strip().lower() == "y"
    return ask("The agent escalated. Continue with another round unchanged (c) or stop (s)? [s] "
               ).strip().lower() == "c"


def _update_decision(folder: Path, index: int, **fields):
    entries = read_jsonl(folder / "decisions.jsonl")
    entries[index].update(fields)
    (folder / "decisions.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")


def judge(before: dict, round_trials: list[dict], outcome: dict) -> tuple[str, dict]:
    """Outcome of the decision before this round: its best run against the best run before it."""
    after = min(round_trials, key=lambda t: t["value"])
    metrics = {"value_before": before["value"], "value_after": after["value"],
               "non_pd_before": before["non_pd_rate"], "non_pd_after": after["non_pd_rate"],
               "best_run_after": after["run_id"]}
    improved = (before["value"] - after["value"] > outcome.get("min_loss_drop", 0.0)
                or before["non_pd_rate"] - after["non_pd_rate"] > outcome.get("min_non_pd_drop", 0.0))
    return ("improved" if improved else "failed"), metrics


def run_loop(loop: dict, name: str, runs_dir: str | Path = "runs", approve: str = "ask",
             graphs=None, decide_fn=None, ask=input, log=print) -> dict:
    """Run the loop until a stopping rule or an escalation; returns summary.json's content."""
    if approve not in APPROVE_MODES:
        raise ValueError(f"approve must be one of {APPROVE_MODES}")
    folder = Path(runs_dir) / name
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "loop.json").write_text(json.dumps(
        {"loop": loop, "approve": approve, "git_commit": git_commit(), "optuna": optuna.__version__}, indent=1))
    if graphs is None:
        graphs = load_dataset(loop["data"]["paths"])
    if decide_fn is None:
        claude = shutil.which("claude") or "claude"
        decide_fn = lambda f, lp, structure: decide(f, lp, claude, structure=structure)  # noqa: E731

    budget, objective = loop["budget"], loop["optimizer"]["objective"]
    search = copy.deepcopy(loop["search"])
    structure = {k: v[0] for k, v in loop.get("menu", {}).items()}
    structure.update(loop.get("start_structure") or {})
    trials, best, since_best, pending, round_no = [], None, 0, None, 0

    while True:
        n_runs = min(budget.get("round_runs", budget["runs"]), budget["runs"] - len(trials))
        log(f"round {round_no}: {n_runs} runs, structure {structure}")
        new = run_round(loop, search, structure, folder, round_no, len(trials), n_runs, graphs)
        before = best
        for t in new:
            if best is None or t["value"] < best["value"]:
                best, since_best = t, 0
            else:
                since_best += 1
        trials += new
        (folder / "trials.json").write_text(json.dumps(
            {"objective": objective, "best": {k: best[k] for k in ("number", "run_id", "value")},
             "trials": trials}, indent=1))
        if pending is not None:
            outcome, metrics = judge(before, new, loop.get("outcome", {}))
            _update_decision(folder, pending, outcome=outcome, outcome_metrics=metrics)
            log(f"  previous decision: {outcome} ({objective} {metrics['value_before']:.4f} -> "
                f"{metrics['value_after']:.4f}, non-PD {metrics['non_pd_before']:.3f} -> {metrics['non_pd_after']:.3f})")
            pending = None

        best_eval = json.loads((folder / best["run_id"] / "eval.json").read_text())
        check = check_target(best_eval, loop["thresholds"], len(trials), budget, since_best)
        log(f"  best {objective} {best['value']:.4f} ({best['run_id']}); {check['decision']}: {check['reason']}")
        if check["decision"] == "stop":
            stop = check["reason"]
            break

        entry = decide_fn(folder, {**loop, "search": search}, structure=dict(structure))
        index = len(read_jsonl(folder / "decisions.jsonl")) - 1
        decision = entry["decision"]
        approved = gate(decision, approve, ask)
        _update_decision(folder, index, approved=approved, approval=approve, round=round_no)
        log(f"  agent: {'forced ' if entry['forced'] else ''}{decision['decision']}"
            f" -> {'approved' if approved else 'not approved'}")

        if decision["decision"] == "escalate":
            if not approved:
                stop = "escalated: " + decision["reason"]
                break
        elif decision["decision"] == "retune":
            search.update(copy.deepcopy(decision["ranges"]))
            pending = index
        elif approved:
            option = decision["menu_option"]
            structure[option["name"]] = option["value"]
            pending = index
        round_no += 1

    summary = {"stop": stop, "runs_used": len(trials), "rounds": round_no + 1,
               "best": {k: best[k] for k in ("number", "run_id", "value")},
               "structure": structure, "search": search,
               "decisions": len(read_jsonl(folder / "decisions.jsonl"))}
    (folder / "summary.json").write_text(json.dumps(summary, indent=1))
    log(f"Stopped: {stop}. Best: {best['run_id']}, {objective} {best['value']:.4f}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run the diagnostician loop.")
    parser.add_argument("--config", default="config/loop.yaml")
    parser.add_argument("--name", required=True, help="results go to runs/<name>/")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--approve", choices=APPROVE_MODES, default="ask",
                        help="human gate for structural changes and escalations (see module docstring)")
    parser.add_argument("--runs", type=int, help="override budget.runs (e.g. a tiny test loop)")
    parser.add_argument("--epochs", type=int, help="override training.epochs")
    args = parser.parse_args()

    loop = yaml.safe_load(Path(args.config).read_text())
    if args.runs is not None:
        loop["budget"]["runs"] = args.runs
    if args.epochs is not None:
        loop["training"]["epochs"] = args.epochs
    if shutil.which("claude") is None:
        sys.exit("Claude Code (`claude`) not found on PATH.")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    run_loop(loop, args.name, args.runs_dir, args.approve)


if __name__ == "__main__":
    main()
