# lattice-gnn
This contains the code used to develop a GNN to predict the elastic tensor of lattice structures.

## GNN pipeline and optimizer baseline

Setup: `python -m venv .venv`, then `.venv/Scripts/python -m pip install -r requirements.txt`.

- Subset: `python scripts/make_subset.py Unit_Cell_Catalog/Unit_Cell_Catalog.txt data/raw/subset_2000.txt --n 2000`
  (every k-th complete record of the catalog).
- Locked split: `splits/main.json` (frozen; see `lattice_gnn/splits.py`).
- One training run: `python train.py data/raw/subset_2000.txt --split-file splits/main.json --epochs 100 --run-id <id>`
  writes `runs/<id>/` (config, split, model, history, `eval.json`).
- Evaluate on validation (`eval.json`) or, once, on the locked test set: `python -m lattice_gnn.evaluate runs/<id> [--part test]`.
- Optuna baseline: `python -m lattice_gnn.optimize --config config/loop.yaml --study <name>`
  (`--runs 6 --epochs 5` for a quick check).
- Readable report of a run: `python -m lattice_gnn.report runs/<id>`.
- Tests: `python -m pytest tests`.

Design: `docs/agent_design_worksheet.md`; build order: `docs/TODO.md`; running log: `LOG.md`.

## Literature-review agent (Claude Code)

Runs in Claude Code with your Claude sign-in (no API key). Setup, once per machine:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r agents/requirements.txt   # Linux/macOS: .venv/bin/python
claude auth status                                                # must show loggedIn: true
```

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
