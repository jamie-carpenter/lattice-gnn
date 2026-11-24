# lattice-gnn

A graph neural network (GNN) surrogate for the elastic stiffness of architected strut lattices,
and a diagnostician agent that decides, after each round of hyperparameter search, why the
surrogate still fails on unseen lattice families and what to do about it: retune the search,
change the model's structure from a fixed menu, or escalate to a human. Code runs every tool
(training, evaluation, the Optuna optimizer, checks); the agent only makes that one decision.

- Agent model: `claude-sonnet-5-5` (pinned in `config/*.yaml`, called through `claude -p` with no
  tools). Literature agent: same model.
- Design: `docs/agent_design_worksheet.md`; build order: `docs/TODO.md`; running log: `LOG.md`;
  lab reports: `reports/`.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt      # Linux/macOS: .venv/bin/python
claude auth status                                           # agent runs need loggedIn: true
```

The agent runs on a Claude Code sign-in (no API key). If `claude auth status` shows
`loggedIn: false`, start `claude` and type `/login`. A failed agent call stops the loop with an
error; it is never recorded as a decision.

## Data, split and one training run

- Subset: `python scripts/make_subset.py Unit_Cell_Catalog/Unit_Cell_Catalog.txt data/raw/subset_2000.txt --n 2000`
  (every k-th complete record of the catalog).
- Locked split by lattice family: `splits/main.json` (frozen; see `lattice_gnn/splits.py`). The
  test families are never used during development.
- One training run: `python train.py data/raw/subset_2000.txt --split-file splits/main.json --epochs 100 --run-id <id>`
  writes `runs/<id>/`: `config.json`, `split.json`, `model.pt`, `history.json` and `eval.json`
  (validation errors per target and per family, the positive-definiteness check, and the run's
  CPU- and wall-hours).
- Evaluate on validation, or once on the locked test set: `python -m lattice_gnn.evaluate runs/<id> [--part test]`.
- Readable report of a run: `python -m lattice_gnn.report runs/<id>`.

## The diagnostician loop

```bash
python -m agents.diagnostician.loop --config config/loop_lr4.yaml --name <name> --approve all
```

Each round: Optuna trains `budget.round_runs` models over the current search ranges ->
evaluation -> stopping check -> agent decision -> validator (`lattice_gnn/validate.py`) ->
human gate (`--approve ask | all | none`) -> apply. The next round's outcome is judged by code
against thresholds set from the seed-to-seed spread. Results in `runs/<name>/`: `trials.json`,
`decisions.jsonl` (each decision, its validation and outcome), `agent/` (every prompt and reply),
`run_info.json` (model, tokens, cost per call), `summary.json`.

- Baselines with the same tools and budget: the rule table (`--decider rules`,
  `agents/diagnostician/rules.py`) and Optuna over hyperparameters and menu together
  (`python -m lattice_gnn.optimize --config <config> --study <name>`).
- One decision on an existing run or study: `python -m agents.diagnostician runs/<folder>`
  (`--prompt-only` writes the agent's input without calling it).
- Quick check: add `--runs 4 --epochs 2` (loop) or `--runs 6 --epochs 5` (Optuna).

Configs: `config/loop.yaml` (full catalog), `config/loop_subset.yaml` (Lab Report 3 study, budget
in runs, width capped at 256), `config/loop_lr4.yaml` (Lab Report 4: budget of 22 CPU-hours,
width 512 allowed). A budget can count training runs (`budget.runs`), CPU-hours
(`budget.cpu_hours`, CPU time summed over threads, so laptop standby does not count), or both.

## Planted failures (Lab Report 4)

Six cases in `config/planted.yaml`, each a fault injected into the Lab Report 4 setup with an
answer key that code checks, fixed before any case was run and never shown to the agent:
`collapse`, `units_low_lr` and `width_trap` have a correct remedy; `control`, `hidden_noise` and
`two_causes` have none, and only unsupported decisions are flagged.

```bash
python -m agents.diagnostician.planted list
bash scripts/run_lr4_study.sh                     # all cases, about a day of laptop time
bash scripts/run_lr4_study.sh collapse            # one case
python -m agents.diagnostician.planted score runs/planted-*
```

Known cases run the agent loop, 10 repeated agent calls on that loop's first-round report (also
the decision-consistency test) and the rule-table loop; the others run one round and then 10
agent calls and the rule table. Progress: `runs/logs/lr4-study.log`. Keep the laptop plugged in:
on battery, Windows Modern Standby pauses training when the screen turns off.

Earlier study scripts: `scripts/run_lr3_study.sh` (Lab Report 3, three conditions on the subset).

## Tests

`python -m pytest tests` (no agent calls: a fake `claude` stands in).

## Literature-review agent (Claude Code)

Support tool, outside the research loop. Setup once: `.venv/Scripts/python -m pip install -r agents/requirements.txt`.

- Reproducible run: `.venv/Scripts/python -m agents.literature [topic] --name <folder>`
  (add `--claude <path to claude.exe>` if `claude` is not on PATH). The agent reaches its tools
  through `python -m agents.literature.cli`, with Bash restricted to that one command.
- Interactive: start `claude` in this folder and type `/literature-review [topic]`. This uses the
  MCP version of the tools; check `/mcp` shows the `literature` server as connected
  (`--via mcp` runs the same version non-interactively).
- Output: `literature_review/<folder>/` with `papers.json`, `lit_review.md`, `run_info.json`.

Agent definitions: `.claude/agents/literature-reviewer-cli.md` (default) and
`.claude/agents/literature-reviewer.md` (MCP); tools: `agents/literature/server.py`, called directly
by `agents/literature/cli.py` or as an MCP server registered in `.mcp.json`.

On the course WorkSpace, install Python 3.12 from the NuGet package (MSI installers are blocked):
download https://www.nuget.org/api/v2/package/python/3.12.10, unzip it, and put its `tools`
folder on your user PATH.
