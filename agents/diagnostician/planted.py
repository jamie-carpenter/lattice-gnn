"""Planted failures (Lab Report 4): run each case of config/planted.yaml and score the decisions.

A case is config/loop_lr4.yaml with the case's `overrides` merged in. Its answer key stays in
config/planted.yaml and in <folder>/planted.json; it never enters the agent's input (the loop
builds that from the loop settings, the eval report, the history and the decision log only).

Modes (repo root, laptop plugged in):
    python -m agents.diagnostician.planted list
    python -m agents.diagnostician.planted loop   --case divergence --decider agent
    python -m agents.diagnostician.planted frozen --case control --repeats 10
    python -m agents.diagnostician.planted repeat --from runs/planted-divergence-agent --repeats 10
    python -m agents.diagnostician.planted score  runs/planted-*

loop    the full loop (--decider agent | rules, --approve all), folder runs/planted-<case>-<decider>
frozen  one round of training (round 0), then --repeats agent calls and one rule-table call, each
        on the same frozen report as if it were the first decision; folder runs/planted-<case>-frozen
repeat  the same calls on round 0 of an existing loop folder (copied, so the loop stays as it is);
        folder <from>-repeats
score   per folder: each check of the answer key (known cases) or each flagged decision
        (ambiguous cases), as a Markdown table; --json for the raw result

Frozen calls are independent: each starts with an empty decision log, and its entry is moved to
repeats.jsonl afterwards.
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
from lattice_gnn.training import git_commit

from .agent import decide, read_jsonl
from .loop import run_loop, run_round
from .rules import decide_rules

PLANTED = Path("config/planted.yaml")


# ---------------------------------------------------------------- building a case

def merge(base: dict, over: dict) -> dict:
    """Dicts merged key by key; anything else in `over` replaces the base value."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_cases(path: str | Path = PLANTED) -> tuple[dict, dict]:
    spec = yaml.safe_load(Path(path).read_text())
    base = yaml.safe_load(Path(spec["base"]).read_text())
    return base, spec["cases"]


def case_loop(base: dict, case: dict) -> dict:
    loop = merge(base, case.get("overrides", {}))
    loop.pop("planted", None)
    return loop


def start_structure(loop: dict) -> dict:
    structure = {k: v[0] for k, v in loop.get("menu", {}).items()}
    structure.update(loop.get("start_structure") or {})
    return structure


def write_key(folder: Path, name: str, case: dict, mode: str):
    (folder / "planted.json").write_text(json.dumps(
        {"case": name, "kind": case["kind"], "mode": mode, "description": case["description"],
         "answer": case["answer"], "git_commit": git_commit()}, indent=1))


# ---------------------------------------------------------------- frozen repeats

def frozen_calls(folder: Path, loop: dict, structure: dict, repeats: int, decide_fn=None,
                 rules: bool = True, log=print) -> list[dict]:
    """`repeats` agent calls (and one rule-table call) on the frozen report in `folder`, each as
    a first decision. Entries go to repeats.jsonl with `decider` and `repeat` set."""
    if decide_fn is None:
        claude = shutil.which("claude") or "claude"
        decide_fn = lambda f, lp, s: decide(f, lp, claude, structure=s)  # noqa: E731
    out = []
    jobs = [("agent", i) for i in range(repeats)] + ([("rules", 0)] if rules else [])
    for decider, i in jobs:
        log_file = folder / "decisions.jsonl"
        if log_file.exists():
            log_file.unlink()
        fn = decide_fn if decider == "agent" else (lambda f, lp, s: decide_rules(f, lp, s))
        entry = fn(folder, copy.deepcopy(loop), dict(structure))
        entry = {**entry, "decider": decider, "repeat": i}
        log_file.unlink()
        with open(folder / "repeats.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        log(f"  {decider} {i}: {'forced ' if entry['forced'] else ''}{entry['decision']['decision']}")
        out.append(entry)
    return out


def train_round0(loop: dict, folder: Path, graphs) -> list[dict]:
    """Round 0 of the loop alone: `round_runs` runs over the starting ranges and structure."""
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "loop.json").write_text(json.dumps(
        {"loop": loop, "mode": "frozen", "git_commit": git_commit(), "optuna": optuna.__version__}, indent=1))
    trials = run_round(loop, copy.deepcopy(loop["search"]), start_structure(loop), folder, 0, 0,
                       loop["budget"]["round_runs"], graphs)
    write_trials(folder, loop, trials)
    return trials


def write_trials(folder: Path, loop: dict, trials: list[dict]):
    best = min(trials, key=lambda t: t["value"])
    (folder / "trials.json").write_text(json.dumps(
        {"objective": loop["optimizer"]["objective"],
         "best": {k: best[k] for k in ("number", "run_id", "value")},
         "cpu_hours_used": sum(t.get("cpu_hours", 0.0) for t in trials), "trials": trials}, indent=1))


def copy_round0(src: Path, dst: Path) -> tuple[dict, dict]:
    """Copy round 0 of a finished loop folder for repeats; returns (loop settings, planted key)."""
    dst.mkdir(parents=True, exist_ok=False)
    shutil.copytree(src / "round-0", dst / "round-0")
    loop = json.loads((src / "loop.json").read_text())["loop"]
    trials = [t for t in json.loads((src / "trials.json").read_text())["trials"] if t["round"] == 0]
    write_trials(dst, loop, trials)
    (dst / "loop.json").write_text(json.dumps({"loop": loop, "mode": "repeat", "from": str(src)}, indent=1))
    key = json.loads((src / "planted.json").read_text())
    return loop, key


# ---------------------------------------------------------------- scoring (no LLM)

def _text(decision: dict) -> str:
    parts = [decision.get("reason", ""), decision.get("would_try", ""), decision.get("prediction", "")]
    return " ".join(parts + list(decision.get("evidence", []))).lower()


def matches(decision: dict, match: dict) -> bool:
    """Does one decision (the agent's JSON answer) satisfy every key of `match`?"""
    if "decision" in match and decision.get("decision") != match["decision"]:
        return False
    ranges = decision.get("ranges") or {}
    lr = ranges.get("lr") or {}
    if "lr_high_max" in match and not (lr.get("high") is not None and lr["high"] <= match["lr_high_max"]):
        return False
    if "lr_low_min" in match and not (lr.get("low") is not None and lr["low"] >= match["lr_low_min"]):
        return False
    for setting, value in (match.get("excludes") or {}).items():
        spec = ranges.get(setting)
        if not spec or "choices" not in spec or value in spec["choices"]:
            return False
    if "menu_option" in match:
        option = decision.get("menu_option") or {}
        if option.get("name") != match["menu_option"]["name"] or option.get("value") != match["menu_option"]["value"]:
            return False
    if "mentions" in match and match["mentions"].lower() not in _text(decision):
        return False
    return True


def check_passes(check: dict, decisions: list[dict]) -> bool:
    scope = decisions[:1] if check.get("when", "any") == "first" else decisions
    alternatives = check.get("any_of") or [check["match"]]
    return any(matches(d, m) for d in scope for m in alternatives)


def score_sequence(answer: dict, kind: str, decisions: list[dict]) -> dict:
    """One decision sequence (a loop's log, or a single frozen call) against the answer key."""
    if kind == "known":
        return {"checks": {c["id"]: check_passes(c, decisions) for c in answer.get("checks", [])},
                "wrong": {c["id"]: check_passes(c, decisions) for c in answer.get("wrong", [])}}
    first = decisions[:1]
    return {"first": first[0]["decision"] if first else None,
            "unsupported": any(matches(d, m) for d in first for m in answer.get("unsupported", []))}


def score_folder(folder: Path) -> dict:
    key = json.loads((folder / "planted.json").read_text())
    result = {"folder": str(folder), "case": key["case"], "kind": key["kind"], "mode": key["mode"]}
    summary = folder / "summary.json"
    if summary.exists():
        s = json.loads(summary.read_text())
        result.update(stop=s["stop"], runs_used=s["runs_used"], cpu_hours_used=s.get("cpu_hours_used"),
                      best_value=s["best"]["value"])
    loop_log = read_jsonl(folder / "decisions.jsonl")
    if loop_log:
        decisions = [e["decision"] for e in loop_log]
        result["loop"] = {"decisions": [d["decision"] for d in decisions],
                          "forced": [e["forced"] for e in loop_log],
                          **score_sequence(key["answer"], key["kind"], decisions)}
    repeats = read_jsonl(folder / "repeats.jsonl")
    if repeats:
        result["repeats"] = [{"decider": e["decider"], "repeat": e["repeat"], "forced": e["forced"],
                              **score_sequence(key["answer"], key["kind"], [e["decision"]])}
                             for e in repeats]
    return result


def _row_known(r: dict, label: str, s: dict) -> str:
    checks = ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in s["checks"].items()) or "-"
    wrong = ", ".join(f"{k} {'YES' if v else 'no'}" for k, v in s["wrong"].items()) or "-"
    return f"| {r['case']} | {label} | {checks} | {wrong} |"


def markdown(results: list[dict]) -> str:
    lines = ["| Case | Run | Checks passed | Wrong answers |", "|---|---|---|---|"]
    amb = ["| Case | Run | First decision | Unsupported |", "|---|---|---|---|"]
    for r in results:
        if "loop" in r:
            label = f"loop ({' > '.join(r['loop']['decisions']) or 'no decision'}; {r.get('stop', '')})"
            if r["kind"] == "known":
                lines.append(_row_known(r, label, r["loop"]))
            else:
                amb.append(f"| {r['case']} | {label} | {r['loop']['first']} | {r['loop']['unsupported']} |")
        for decider in ("agent", "rules"):
            reps = [x for x in r.get("repeats", []) if x["decider"] == decider]
            if not reps:
                continue
            label = f"frozen, {decider} x{len(reps)}"
            if r["kind"] == "known":
                checks = {k: sum(x["checks"][k] for x in reps) for k in reps[0]["checks"]}
                wrong = {k: sum(x["wrong"][k] for x in reps) for k in reps[0]["wrong"]}
                lines.append(f"| {r['case']} | {label} | "
                             + (", ".join(f"{k} {v}/{len(reps)}" for k, v in checks.items()) or "-") + " | "
                             + (", ".join(f"{k} {v}/{len(reps)}" for k, v in wrong.items()) or "-") + " |")
            else:
                spread = {}
                for x in reps:
                    spread[x["first"]] = spread.get(x["first"], 0) + 1
                amb.append(f"| {r['case']} | {label} | "
                           + ", ".join(f"{k} {v}" for k, v in sorted(spread.items())) + " | "
                           + f"{sum(x['unsupported'] for x in reps)}/{len(reps)} |")
    out = []
    if len(lines) > 2:
        out += ["Known cases", "", *lines, ""]
    if len(amb) > 2:
        out += ["Ambiguous cases", "", *amb, ""]
    return "\n".join(out)


# ---------------------------------------------------------------- command line

def main():
    parser = argparse.ArgumentParser(description="Planted failures: run cases and score them.")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("list")
    p_loop = sub.add_parser("loop")
    p_loop.add_argument("--case", required=True)
    p_loop.add_argument("--decider", choices=("agent", "rules"), required=True)
    p_fro = sub.add_parser("frozen")
    p_fro.add_argument("--case", required=True)
    p_rep = sub.add_parser("repeat")
    p_rep.add_argument("--from", dest="src", required=True)
    for p in (p_fro, p_rep):
        p.add_argument("--repeats", type=int, default=10)
    for p in (p_loop, p_fro):
        p.add_argument("--name", help="folder name under --runs-dir (default from the case)")
        p.add_argument("--epochs", type=int, help="override training.epochs (smoke tests)")
        p.add_argument("--cpu-hours", type=float, help="override budget.cpu_hours (smoke tests)")
        p.add_argument("--runs-dir", default="runs")
    p_sc = sub.add_parser("score")
    p_sc.add_argument("folders", nargs="+")
    p_sc.add_argument("--json", action="store_true")
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if args.mode == "score":
        results = [score_folder(Path(f)) for f in args.folders if (Path(f) / "planted.json").exists()]
        print(json.dumps(results, indent=1) if args.json else markdown(results))
        return
    base, cases = load_cases()
    if args.mode == "list":
        for name, case in cases.items():
            print(f"{name:14s} {case['kind']:9s} {' '.join(case['description'].split())}")
        return
    if args.mode == "repeat":
        src = Path(args.src)
        dst = src.parent / f"{src.name}-repeats"
        loop, key = copy_round0(src, dst)
        (dst / "planted.json").write_text(json.dumps({**key, "mode": "repeat"}, indent=1))
        _need_claude()
        frozen_calls(dst, loop, start_structure(loop), args.repeats)
        return

    if args.case not in cases:
        sys.exit(f"unknown case {args.case!r}; cases: {', '.join(cases)}")
    case = cases[args.case]
    loop = case_loop(base, case)
    if args.epochs is not None:
        loop["training"]["epochs"] = args.epochs
    if args.cpu_hours is not None:
        loop["budget"]["cpu_hours"] = args.cpu_hours
    runs_dir = Path(args.runs_dir)
    graphs = load_dataset(loop["data"]["paths"])
    if args.mode == "loop":
        if case["kind"] != "known":
            sys.exit(f"{args.case} is an ambiguous case: run it with `frozen`")
        name = args.name or f"planted-{args.case}-{args.decider}"
        if args.decider == "agent":
            _need_claude()
        decide_fn = (lambda f, lp, structure: decide_rules(f, lp, structure)) if args.decider == "rules" else None
        # planted.json is written once the loop has made its folder, before the first decision
        run_loop(loop, name, runs_dir, "all", graphs=graphs, decide_fn=_keyed(decide_fn, args.case, case),
                 decider=args.decider)
        if not (runs_dir / name / "planted.json").exists():      # stopped before any decision
            write_key(runs_dir / name, args.case, case, "loop")
        return
    name = args.name or f"planted-{args.case}-frozen"
    _need_claude()
    folder = runs_dir / name
    train_round0(loop, folder, graphs)
    write_key(folder, args.case, case, "frozen")
    frozen_calls(folder, loop, start_structure(loop), args.repeats)


def _keyed(decide_fn, name: str, case: dict):
    """Wrap the decider so planted.json exists in the loop folder (written before the first call)."""
    if decide_fn is None:
        claude = shutil.which("claude") or "claude"
        decide_fn = lambda f, lp, structure: decide(f, lp, claude, structure=structure)  # noqa: E731

    def fn(folder, loop, structure):
        if not (Path(folder) / "planted.json").exists():
            write_key(Path(folder), name, case, "loop")
        return decide_fn(folder, loop, structure)
    return fn


def _need_claude():
    if shutil.which("claude") is None:
        sys.exit("Claude Code (`claude`) not found on PATH.")


if __name__ == "__main__":
    main()
