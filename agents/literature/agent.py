"""Literature-review agent for the lattice-GNN project (Claude Agent SDK).

Run from the repo root:
    python -m agents.literature                       # default topic
    python -m agents.literature "defects in additively manufactured lattices" --out literature_review/defects

Writes <out>/papers.json (structured bibliography for downstream agents) and
<out>/lit_review.md (the written review).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKError,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    query,
)

from dotenv import load_dotenv

from .tools import TOOL_NAMES, build_server

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TOPIC = (
    "Machine-learning surrogates (especially graph neural networks) that predict the effective "
    "elastic stiffness tensor of periodic truss/strut lattice unit cells, plus the datasets and "
    "homogenization methods used to train them, and how manufacturing defects change lattice stiffness."
)

SYSTEM_PROMPT = """\
You are the literature-review agent in a multi-agent system that designs and tunes a neural network
for predicting the effective stiffness tensor of architected lattice unit cells. Your output is read
by the user (a graduate student) and by later agents that will choose the architecture, tune it, and
add manufacturing defects.

## The project
- Dataset (already chosen): text records, one per lattice, with names like `cub_Z06.0_E1` and
  `ort_Z08.0_E2` (crystal system + average connectivity Z), alternative names including RCSR net
  symbols (e.g. `pcu`), normalized cell parameters, nodal positions and bar connectivities, and
  "effective normalized mechanical properties" (Ex, Ey, Ez, Gyz, Gxz, Gxy, six Poisson's ratios)
  plus scaling constants C and exponents n per direction (E/Es = C * rho_rel^n).
  It may be the crystal-network lattice dataset of Lumpe & Stankovic (PNAS, 2021), but VERIFY this
  from the paper itself; do not assume it.
- Units and conventions:
  - Geometry is dimensionless. Node positions are fractional coordinates, cell lengths are
    normalized so the longest is 1, angles are in degrees, and strut lengths are in cell units.
    Strut thickness is not an input.
  - Moduli are dimensionless, normalized by the base material's Young's modulus Es. Solving
    E/Es = C * rho_rel^n across the full catalog (17,221 lattices) gives a median rho_rel of
    1.00% (98% between 0.90% and 1.12%), for both stretching- (n ~ 1) and bending-dominated
    (n ~ 2) lattices, so the listed moduli appear to be at 1% relative density; the scatter fits
    C and n being power-law fits printed to 2 decimals. Confirm this, how C and n were fitted,
    and the strut model used (beam theory, cross-section), in the dataset paper.
- The catalog has 17,262 lattices. 41 are marked "very small members ... excluded from the main
  manuscript" and have no properties. Find out how the authors handled them.
- Model: message-passing GNN on the periodic graph (nodes = strut junctions, edges = struts with
  length/direction features, periodic edges included), updating nodes and struts, mean+max pooling,
  MLP head. Graph-level inputs: Z_avg and log10(total strut length / cell volume).
- Targets (9 orthotropic constants, standardized): log10 of Ex, Ey, Ez, Gyz, Gxz, Gxy, plus three
  Poisson coupling terms (yz, xz, xy). The model predicts engineering constants, not the C_ij
  stiffness matrix directly; the 6x6 orthotropic matrix is assembled from them.
- Later: manufacturing defects (strut-diameter variation, waviness, missing struts, porosity,
  node misplacement) and how they reduce stiffness.

## How to work
1. Plan 6-10 distinct search angles first (surrogates for lattice/metamaterial elasticity; GNNs for
   periodic structures and crystals; datasets; tensor symmetry/equivariance; homogenization methods;
   defects and as-built vs as-designed stiffness; inverse design).
2. Identify the source paper of OUR dataset and record it (category "dataset"). Then use
   get_paper_links(direction="citations") on it to find papers that trained ML models on the same
   data. Their reported errors are our baselines, so note the metric, the split, and whether they
   predict log-moduli, normalized moduli or C_ij.
3. Search with BOTH search_openalex and search_arxiv. Rephrase when results are poor.
4. Pick 3-5 seminal or highly relevant papers and snowball with get_paper_links, using both
   citations and references.
5. For every paper you rate relevance 4-5 (and the dataset paper), read the full text with read_pdf
   using its pdf_url: pages 1-3 first, then the methods/results pages. For the rest, the abstract
   is enough. If no open-access PDF exists, try WebSearch for an arXiv or author copy; otherwise
   mark it "abstract only".
6. Call record_paper for each genuinely relevant paper, with an `evidence` entry (page + location +
   short quote) for every specific claim you will make about it. Aim for 25-40 papers, with
   relevance scores that discriminate (most should not be 5).
7. Finish with write_review. Sections:
   Overview; Our dataset (source, how labels were computed, assumptions such as relative density
   and strut model, known limitations); Other datasets; Model architectures (comparison table:
   paper, representation, model, targets and their units/normalization, data size, reported
   accuracy, page); Baselines on our dataset; Enforcing tensor symmetry and physical constraints;
   Label generation / homogenization; Manufacturing defects; Gaps and recommendations for this
   project; References (include the PDF version read for each paper).

## Rules
- Never cite or record a paper that a tool did not return. Do not invent DOIs, numbers or authors.
- Cite every specific claim with its exact location: [Author Year, p. N] or
  [Author Year, p. N, Table 2], where N is the PDF page from read_pdf, and the evidence entry holds
  the supporting quote. If you only read the abstract, cite [Author Year, abstract]. Never give a
  page number you did not read.
- When comparing reported accuracies, state each paper's units and normalization (absolute GPa,
  E/Es, log scale, relative density). Do not compare numbers across different conventions.
- If a detail is unknown, say so rather than guessing.
"""


def build_options(out_dir: Path, model: str, max_turns: int, budget: float) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        model=model,
        mcp_servers={"literature": build_server(out_dir)},
        # Built-in tools available to the agent: web only. No shell, no file editing;
        # it can only write files through record_paper / write_review into out_dir.
        tools=["WebSearch", "WebFetch"],
        allowed_tools=["WebSearch", "WebFetch"] + [f"mcp__literature__{t}" for t in TOOL_NAMES],
        permission_mode="dontAsk",
        setting_sources=[],  # ignore user/project Claude Code settings so runs are reproducible
        cwd=str(out_dir),
        max_turns=max_turns,
        max_budget_usd=budget,
    )


def _short(d: dict, n: int = 110) -> str:
    s = ", ".join(f"{k}={v!r}" for k, v in d.items() if k not in ("markdown", "summary", "methods",
                                                                 "data", "key_findings", "relevance_to_project",
                                                                 "evidence"))
    return s if len(s) <= n else s[:n] + "…"


async def run(topic: str, out_dir: Path, model: str, max_turns: int, budget: float) -> int:
    options = build_options(out_dir.resolve(), model, max_turns, budget)
    prompt = f"Conduct a literature review on:\n\n{topic}"
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    models_used: set[str] = set()
    exit_code = 1
    # Consume the whole stream (no early return), so the SDK can shut down cleanly.
    async for msg in query(prompt=prompt, options=options):
        if isinstance(msg, AssistantMessage):
            models_used.add(msg.model)
            for block in msg.content:
                if isinstance(block, TextBlock) and block.text.strip():
                    print(f"\n{block.text.strip()}\n", flush=True)
                elif isinstance(block, ToolUseBlock):
                    print(f"  -> {block.name.removeprefix('mcp__literature__')}({_short(block.input)})", flush=True)
        elif isinstance(msg, ResultMessage):
            # API failures (bad key, no credit) arrive as subtype "success" with the cause in terminal_reason.
            failed = msg.is_error or "error" in (msg.terminal_reason or "")
            cost = f"${msg.total_cost_usd:.2f}" if msg.total_cost_usd is not None else "n/a"
            status = f"FAILED: {msg.terminal_reason or msg.subtype}" if failed else msg.subtype
            print(f"\n[{status}] turns={msg.num_turns} cost={cost} "
                  f"time={msg.duration_ms / 1000:.0f}s -> {out_dir}")
            if msg.errors:
                print("errors:", *msg.errors, sep="\n  ")
            # Pin the model ID and run settings beside the results, so the run can be reproduced.
            run_info = {
                "started_utc": started, "topic": topic,
                "model_requested": model, "models_used": sorted(models_used),
                "claude_agent_sdk": version("claude-agent-sdk"),
                "max_turns": max_turns, "budget_usd": budget,
                "outcome": msg.subtype, "terminal_reason": msg.terminal_reason, "turns": msg.num_turns,
                "cost_usd": msg.total_cost_usd, "duration_s": round(msg.duration_ms / 1000),
                "session_id": msg.session_id, "errors": msg.errors,
            }
            (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2), "utf-8")
            exit_code = 1 if failed else 0
    return exit_code


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("topic", nargs="?", default=DEFAULT_TOPIC)
    p.add_argument("--out", type=Path, default=Path("literature_review"))
    p.add_argument("--model", default="claude-sonnet-5-5")
    p.add_argument("--max-turns", type=int, default=120)
    p.add_argument("--budget", type=float, default=10.0, help="stop once API spend reaches this many USD")
    args = p.parse_args()
    sys.stdout.reconfigure(errors="replace")  # Windows consoles can't print every symbol in paper text
    load_dotenv(REPO_ROOT / ".env")  # ANTHROPIC_API_KEY lives in .env (git-ignored), never in code
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_CODE_USE_BEDROCK")):
        sys.exit(f"No model access: put ANTHROPIC_API_KEY=... in {REPO_ROOT / '.env'} "
                 "(or set CLAUDE_CODE_USE_BEDROCK=1 with AWS credentials).")
    try:
        code = asyncio.run(run(args.topic, args.out, args.model, args.max_turns, args.budget))
    except ClaudeSDKError as e:  # raised after a failed result, which run() has already reported
        print(f"error: {e}", file=sys.stderr)
        code = 1
    sys.exit(code)
