"""Command-line access to the literature tools, one tool call per command.

For runs where the agent reaches its tools through a shell command instead of the MCP server
(python -m agents.literature --via cli). The tools are the ones in server.py, called through the
server itself, so arguments are validated the same way and outputs still go only to
literature_review/<name>/ (papers.json, lit_review.md).

Run from the repo root; arguments are a JSON object, given as one argument or on stdin:
    python -m agents.literature.cli search_arxiv --name main '{"query": "ti:lattice", "max_results": 5}'
    python -m agents.literature.cli write_review --name main '{"markdown": "# Review ...", "append": true}'

write_review also takes "append": true, which adds to the existing lit_review.md, so a long review
can be written in parts (a Windows command line holds at most ~32,000 characters).
"""
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS = ("search_arxiv", "search_openalex", "get_paper_links", "read_pdf", "record_paper", "write_review")

# Like launch_server.py: switch to .venv's Python (which has the tools' dependencies) when it exists.
if __name__ == "__main__":
    for venv_python in (REPO_ROOT / ".venv" / "Scripts" / "python.exe", REPO_ROOT / ".venv" / "bin" / "python"):
        if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
            sys.exit(subprocess.call([str(venv_python), "-m", "agents.literature.cli", *sys.argv[1:]],
                                     cwd=REPO_ROOT, env=os.environ))

import argparse  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402


async def call(tool: str, name: str, arguments: dict) -> tuple[bool, str]:
    from agents.literature import server as S

    for step, args in (("start_review", {"name": name}), (tool, arguments)):
        try:
            result = await S.server.call_tool(step, args)
        except Exception as e:  # validation and tool errors
            return False, f"{step} failed: {e}"
        text = "\n".join(c.text for c in getattr(result, "content", []) if getattr(c, "text", None))
        if getattr(result, "isError", False):
            return False, text or f"{step} failed"
    return True, text


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tool", choices=TOOLS)
    p.add_argument("--name", required=True, help="review folder under literature_review/")
    p.add_argument("arguments", nargs="?", help="JSON object of tool arguments (default: read stdin)")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raw = args.arguments if args.arguments is not None else sys.stdin.read()
    try:
        arguments = json.loads(raw or "{}")
    except json.JSONDecodeError as e:
        sys.exit(f"arguments are not valid JSON: {e}")
    if not isinstance(arguments, dict):
        sys.exit("arguments must be a JSON object")
    if args.tool == "write_review" and arguments.pop("append", False):
        from agents.literature import server as S
        existing = S.OUT_ROOT / args.name / "lit_review.md"
        if existing.exists() and isinstance(arguments.get("markdown"), str):
            arguments["markdown"] = existing.read_text("utf-8").rstrip("\n") + "\n\n" + arguments["markdown"]
    ok, text = asyncio.run(call(args.tool, args.name, arguments))
    print(text)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
