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
import sys
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    query,
)

from .tools import TOOL_NAMES, build_server

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
- Input: a periodic strut lattice unit cell, represented as a graph (nodes = strut junctions,
  edges = struts with length and direction features; periodic boundary edges included).
- Model: message-passing GNN updating both nodes and struts, mean+max pooling, MLP head.
- Output: 9 orthotropic stiffness components (C11, C22, C33, C12, C13, C23, C44, C55, C66),
  standardized; labels come from homogenization.
- Later: imperfections such as strut-diameter variation, waviness, missing struts, porosity,
  and node misplacement, and their effect on stiffness.

## How to work
1. Plan 6-10 distinct search angles first (surrogates for lattice/metamaterial elasticity; GNNs for
   periodic structures and crystals; datasets; tensor symmetry/equivariance; homogenization methods;
   defects and as-built vs as-designed stiffness; inverse design).
2. Search with BOTH search_openalex and search_arxiv. Rephrase when results are poor.
3. Pick 3-5 seminal or highly relevant papers and snowball with get_paper_links, using both
   citations and references.
4. Use WebFetch on a paper page only when the abstract is too thin to judge methods or results.
   Use WebSearch for things the paper APIs miss, such as dataset repositories or code.
5. Call record_paper for each paper that is genuinely relevant. Aim for 25-40 papers, with relevance
   scores that discriminate (most should not be 5).
6. Finish with write_review. Sections:
   Overview; Datasets; Model architectures (with a comparison table: paper, representation, model,
   targets, data size, reported accuracy); Enforcing tensor symmetry and physical constraints;
   Label generation / homogenization; Manufacturing defects; Gaps and recommendations for this
   project; References.

## Rules
- Never cite or record a paper that a tool did not return. Do not invent DOIs, numbers or authors.
- If a detail is unknown, say so rather than guessing.
- Keep claims attributable: every factual claim in the review cites a recorded paper.
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
                                                                 "data", "key_findings", "relevance_to_project"))
    return s if len(s) <= n else s[:n] + "…"


async def run(topic: str, out_dir: Path, model: str, max_turns: int, budget: float) -> int:
    options = build_options(out_dir.resolve(), model, max_turns, budget)
    prompt = f"Conduct a literature review on:\n\n{topic}"
    async for msg in query(prompt=prompt, options=options):
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock) and block.text.strip():
                    print(f"\n{block.text.strip()}\n", flush=True)
                elif isinstance(block, ToolUseBlock):
                    print(f"  -> {block.name.removeprefix('mcp__literature__')}({_short(block.input)})", flush=True)
        elif isinstance(msg, ResultMessage):
            cost = f"${msg.total_cost_usd:.2f}" if msg.total_cost_usd is not None else "n/a"
            print(f"\n[{msg.subtype}] turns={msg.num_turns} cost={cost} "
                  f"time={msg.duration_ms / 1000:.0f}s -> {out_dir}")
            if msg.errors:
                print("errors:", *msg.errors, sep="\n  ")
            return 1 if msg.is_error else 0
    return 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("topic", nargs="?", default=DEFAULT_TOPIC)
    p.add_argument("--out", type=Path, default=Path("literature_review"))
    p.add_argument("--model", default="claude-sonnet-5-5")
    p.add_argument("--max-turns", type=int, default=120)
    p.add_argument("--budget", type=float, default=5.0, help="stop once API spend reaches this many USD")
    args = p.parse_args()
    sys.exit(asyncio.run(run(args.topic, args.out, args.model, args.max_turns, args.budget)))
