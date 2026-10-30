"""Optimizer baseline: an Optuna study over the hyperparameters and the structural menu.

This is the no-agent comparison (agent_design_worksheet.md section 8): same tools, same
budget, same split, every menu option available as a categorical setting.

Usage:
    python -m lattice_gnn.optimize --config config/loop.yaml --study optuna-0
    python -m lattice_gnn.optimize --config config/loop.yaml --study tiny --runs 6 --epochs 5

Writes runs/<study>/: study.json (settings), trials.json (one entry per training run, updated
after each), and one run folder per trial (runs/<study>/trial-000/, ...).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import optuna
import yaml

from .data import load_dataset
from .training import git_commit, train

OBJECTIVES = ("loss", "median_percent_error_moduli", "median_abs_error_couplings", "non_pd_rate")


def score(result: dict, name: str) -> float:
    """One number from an eval.json report; lower is better."""
    if name == "loss":
        return result["loss"]
    if name == "non_pd_rate":
        return result["pd"]["non_pd_rate"]
    return result["overall"][name]


def check_target(result: dict, thresholds: dict, used: int, budget: dict,
                 since_best: int = 0) -> dict:
    """continue | stop, with the reason. `result` is the best run's eval.json so far.

    Stops when every threshold is met (success), when the budget of training runs is used
    up, or after budget["no_progress"] runs in a row without a new best.
    """
    missed = [k for k, limit in thresholds.items() if score(result, k) > limit]
    if not missed:
        return {"decision": "stop", "reason": "target met"}
    if used >= budget["runs"]:
        return {"decision": "stop", "reason": f"budget used ({used} runs)"}
    if since_best >= budget.get("no_progress", math.inf):
        return {"decision": "stop", "reason": f"no progress in {since_best} runs"}
    return {"decision": "continue", "reason": "missed: " + ", ".join(missed)}


def suggest(trial: optuna.Trial, search: dict, menu: dict) -> dict:
    params = {}
    for name, spec in search.items():
        kind = spec["type"]
        if kind == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
        elif kind == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False))
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"search.{name}: unknown type {kind!r}")
    for name, choices in menu.items():
        params[name] = trial.suggest_categorical(name, choices)
    return params


def optimize(loop: dict, study_name: str, runs_dir: str | Path = "runs", graphs=None) -> dict:
    """Run one study; returns the final check_target decision and the best trial."""
    objective_name = loop["optimizer"]["objective"]
    if objective_name not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    study_dir = Path(runs_dir) / study_name
    study_dir.mkdir(parents=True, exist_ok=False)
    (study_dir / "study.json").write_text(json.dumps(
        {"loop": loop, "git_commit": git_commit(), "optuna": optuna.__version__}, indent=1))
    if graphs is None:
        graphs = load_dataset(loop["data"]["paths"])

    sampler = optuna.samplers.TPESampler(seed=loop["optimizer"]["sampler_seed"])
    study = optuna.create_study(study_name=study_name, direction="minimize", sampler=sampler)
    trials, best, state = [], None, {"since_best": 0, "check": None}

    def objective(trial: optuna.Trial) -> float:
        nonlocal best
        params = suggest(trial, loop["search"], loop.get("menu", {}))
        config = {**loop["training"], **params, "paths": loop["data"]["paths"],
                  "split_file": loop["data"]["split_file"], "runs_dir": str(study_dir),
                  "run_id": f"trial-{trial.number:03d}", "verbose": False}
        run_id = train(config, graphs=graphs)
        result = json.loads((study_dir / run_id / "eval.json").read_text())
        value = score(result, objective_name)

        if best is None or value < best["value"]:
            best = {"number": trial.number, "run_id": run_id, "value": value, "eval": result}
            state["since_best"] = 0
        else:
            state["since_best"] += 1
        check = check_target(best["eval"], loop["thresholds"], trial.number + 1, loop["budget"],
                             state["since_best"])
        state["check"] = check
        trials.append({"number": trial.number, "run_id": run_id, "params": params, "value": value,
                       "best_epoch": result["best_epoch"], "overall": result["overall"],
                       "non_pd_rate": result["pd"]["non_pd_rate"], "check": check})
        (study_dir / "trials.json").write_text(json.dumps(
            {"objective": objective_name, "best": {k: best[k] for k in ("number", "run_id", "value")},
             "trials": trials}, indent=1))
        print(f"trial {trial.number:3d}  {objective_name} {value:.4f}  best {best['value']:.4f}"
              f" (trial {best['number']})  -> {check['decision']}: {check['reason']}")
        if check["decision"] == "stop":
            trial.study.stop()
        return value

    study.optimize(objective, n_trials=loop["budget"]["runs"])
    return {"check": state["check"], "best": {k: best[k] for k in ("number", "run_id", "value")}}


def main():
    parser = argparse.ArgumentParser(description="Run the Optuna baseline study.")
    parser.add_argument("--config", default="config/loop.yaml")
    parser.add_argument("--study", required=True, help="name; results go to runs/<study>/")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--runs", type=int, help="override budget.runs (e.g. a tiny test study)")
    parser.add_argument("--epochs", type=int, help="override training.epochs")
    args = parser.parse_args()

    loop = yaml.safe_load(Path(args.config).read_text())
    if args.runs is not None:
        loop["budget"]["runs"] = args.runs
    if args.epochs is not None:
        loop["training"]["epochs"] = args.epochs
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    final = optimize(loop, args.study, args.runs_dir)
    print(f"\nStopped: {final['check']['reason']}. Best: trial {final['best']['number']} "
          f"({final['best']['run_id']}), value {final['best']['value']:.4f}")


if __name__ == "__main__":
    main()
