"""Start the literature tool server with the project's virtual environment, on Windows or Linux.

.mcp.json runs this with whatever `python` is on PATH; it switches to .venv's Python (which has
the server's dependencies) when that exists. Standard library only, on purpose.
"""
import os
import subprocess
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
server = Path(__file__).with_name("server.py")
for venv_python in (repo / ".venv" / "Scripts" / "python.exe", repo / ".venv" / "bin" / "python"):
    if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
        sys.exit(subprocess.call([str(venv_python), str(server)], env=os.environ))

sys.argv = [str(server)]
exec(compile(server.read_text("utf-8"), str(server), "exec"), {"__name__": "__main__", "__file__": str(server)})
