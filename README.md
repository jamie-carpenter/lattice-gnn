# lattice-gnn
This contains the code used to develop a GNN to predict the elastic tensor of lattice structures.

## Literature-review agent (Claude Code)

Runs in Claude Code with your Claude sign-in (no API key). Setup, once per machine:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r agents/requirements.txt   # Linux/macOS: .venv/bin/python
claude auth status                                                # must show loggedIn: true
```

- Interactive: start `claude` in this folder and type `/literature-review [topic]`.
  Check `/mcp` shows the `literature` server as connected.
- Reproducible run: `.venv/Scripts/python -m agents.literature [topic] --name <folder>`
- Output: `literature_review/<folder>/` with `papers.json`, `lit_review.md`, `run_info.json`.

Agent definition: `.claude/agents/literature-reviewer.md`; tools: `agents/literature/server.py`
(registered in `.mcp.json`).
