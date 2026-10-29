"""Run the literature-review agent non-interactively through Claude Code (`claude -p`).

Uses your Claude Code sign-in (the course's Claude Enterprise seat), so there is no API key.
The agent is defined in .claude/agents/literature-reviewer-cli.md (the MCP version in
literature-reviewer.md) and its tools in server.py, reached through cli.py by default.
For an interactive run instead, start `claude` in the repo and type: /literature-review [topic]

Run from the repo root:
    python -m agents.literature                                  # default topic -> literature_review/main/
    python -m agents.literature "defects in AM strut lattices" --name defects
    python -m agents.literature --via mcp                        # tools via the MCP server instead of cli.py

Writes literature_review/<name>/: papers.json, lit_review.md, and run_info.json (pinned model ID,
Claude Code version, turns, outcome), so each run can be audited and repeated.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER = Path(__file__).with_name("server.py")
DEFAULT_MODEL = "claude-sonnet-5-5"
CLI_COMMAND = "python -m agents.literature.cli"


def _short(d: dict, n: int = 110) -> str:
    skip = ("markdown", "summary", "methods", "data", "key_findings", "relevance_to_project", "evidence")
    s = ", ".join(f"{k}={v!r}" for k, v in d.items() if k not in skip)
    return s if len(s) <= n else s[:n] + "…"


def build_command(claude: str, topic: str | None, name: str, model: str, max_turns: int,
                  via: str = "cli") -> list[str]:
    prompt = f"Folder name: {name}\nTopic: {topic or '(none given: use your default topic)'}\n" \
             "Conduct the literature review."
    if via == "mcp":
        # Use this interpreter for the tool server, so the run does not depend on PATH or an active venv.
        mcp_config = {"mcpServers": {"literature": {"type": "stdio", "command": sys.executable, "args": [str(SERVER)]}}}
        agent, tools = "literature-reviewer", ["mcp__literature__*"]
    else:
        # Same tools through agents/literature/cli.py; Bash may run that one command and nothing else.
        mcp_config = {"mcpServers": {}}
        agent, tools = "literature-reviewer-cli", [f"Bash({CLI_COMMAND} *)"]
    return [
        claude, "-p", prompt,
        "--agent", agent,                          # tools + instructions from .claude/agents/
        "--model", model,                          # pinned model ID
        "--max-turns", str(max_turns),
        "--output-format", "stream-json", "--verbose",
        "--mcp-config", json.dumps(mcp_config), "--strict-mcp-config",
        # Fixed tool surface: anything not listed here is denied, never prompted.
        "--permission-mode", "dontAsk",
        "--allowedTools", "WebSearch", "WebFetch", *tools,
    ]


def run(claude: str, topic: str | None, name: str, model: str, max_turns: int, via: str = "cli") -> int:
    out_dir = REPO_ROOT / "literature_review" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    version = subprocess.run([claude, "--version"], capture_output=True, text=True).stdout.strip()
    info: dict = {
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "topic": topic, "name": name, "model_requested": model, "max_turns": max_turns,
        "tools_via": via, "claude_code_version": version,
    }
    proc = subprocess.Popen(build_command(claude, topic, name, model, max_turns, via), cwd=REPO_ROOT,
                            stdout=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    result = None
    for line in proc.stdout:
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("type") == "system" and msg.get("subtype") == "init":
            info["model_used"] = msg.get("model")
            info["mcp_servers"] = msg.get("mcp_servers")
            servers = {s.get("name"): s.get("status") for s in msg.get("mcp_servers") or []}
            if via == "mcp" and servers.get("literature") != "connected":
                print(f"WARNING: literature tool server not connected: {servers}", flush=True)
        elif msg.get("type") == "assistant":
            for block in msg.get("message", {}).get("content", []):
                if block.get("type") == "text" and block.get("text", "").strip():
                    print(f"\n{block['text'].strip()}\n", flush=True)
                elif block.get("type") == "tool_use":
                    tool = block.get("name", "").removeprefix("mcp__literature__")
                    if tool == "Bash":
                        line1 = (block.get("input") or {}).get("command", "").splitlines()[:1]
                        print(f"  -> Bash: {line1[0] if line1 else ''}", flush=True)
                    else:
                        print(f"  -> {tool}({_short(block.get('input') or {})})", flush=True)
        elif msg.get("type") == "result":
            result = msg
    proc.wait()

    if result is None:
        print(f"Claude Code exited (code {proc.returncode}) without a result. "
              "Is it installed and signed in? Try: claude auth status", file=sys.stderr)
        return 1
    failed = bool(result.get("is_error")) or "error" in (result.get("terminal_reason") or "")
    info.update({
        "outcome": result.get("subtype"), "terminal_reason": result.get("terminal_reason"),
        "is_error": failed, "turns": result.get("num_turns"),
        "duration_s": round((result.get("duration_ms") or 0) / 1000), "session_id": result.get("session_id"),
        # Client-side estimate only; usage is covered by the Claude Code seat, not billed per token.
        "cost_estimate_usd": result.get("total_cost_usd"),
        "final_message": result.get("result"),
    })
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), "utf-8")
    status = f"FAILED: {result.get('terminal_reason') or result.get('subtype')}" if failed else result.get("subtype")
    print(f"\n[{status}] turns={info['turns']} time={info['duration_s']}s -> {out_dir}")
    return 1 if failed else 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("topic", nargs="?", help="omit to use the agent's default topic")
    p.add_argument("--name", default="main", help="output folder under literature_review/ (default: main)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-turns", type=int, default=120)
    p.add_argument("--claude", help="path to the claude executable (default: found on PATH)")
    p.add_argument("--via", choices=("cli", "mcp"), default="cli",
                   help="reach the tools through the command line (cli.py, default; needs only the venv) "
                        "or the MCP server (needs Claude Code to load project MCP servers)")
    args = p.parse_args()
    sys.stdout.reconfigure(errors="replace")  # Windows consoles can't print every symbol in paper text
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,60}", args.name):
        sys.exit("--name must be lowercase letters, digits, '-' or '_'")
    claude = args.claude or shutil.which("claude")
    if not claude:
        sys.exit("Claude Code ('claude') was not found on PATH. Install it and sign in with your "
                 "Caltech account (claude auth login), or pass --claude <path>.")
    sys.exit(run(claude, args.topic, args.name, args.model, args.max_turns, args.via))
