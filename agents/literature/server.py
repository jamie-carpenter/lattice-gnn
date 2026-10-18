"""MCP tool server for the literature-review agent (Claude Code connects to it over stdio).

Registered for Claude Code in the repo's .mcp.json as server "literature", so its tools appear
as mcp__literature__<name>. The agent itself (instructions, allowed tools, model) is defined in
.claude/agents/literature-reviewer.md.

Outputs go to literature_review/<name>/ in the repo (override the root with LIT_REVIEW_ROOT):
papers.json (structured bibliography for downstream agents) and lit_review.md (the review).
The agent can write nowhere else: these two files are its only write access.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))  # so this file also runs as a plain script

from mcp.server.mcpserver import MCPServer  # noqa: E402

from agents.literature import tools as T  # noqa: E402

OUT_ROOT = Path(os.environ.get("LIT_REVIEW_ROOT", REPO_ROOT / "literature_review"))
NAME_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,60}")

server = MCPServer(
    name="literature",
    instructions="Literature search, open-access PDF reading, and bibliography tools. "
                 "Call start_review first; it chooses the output folder.",
)

_state: dict[str, Any] = {"dir": None, "papers": []}


def _json(rows: list[dict[str, Any]]) -> str:
    return json.dumps(rows, indent=1, ensure_ascii=False)


def _out_dir() -> Path:
    if _state["dir"] is None:
        raise ValueError("Call start_review(name) first to choose the output folder.")
    return _state["dir"]


@server.tool()
async def start_review(
    name: Annotated[str, Field(description="Folder name for this review: lowercase letters, digits, - or _")],
) -> str:
    """Start (or resume) a review. Outputs go to literature_review/<name>/; existing papers.json is kept."""
    if not NAME_RE.fullmatch(name):
        raise ValueError("name must be lowercase letters, digits, '-' or '_' (max 61 chars)")
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    papers_path = out / "papers.json"
    _state["dir"] = out
    _state["papers"] = json.loads(papers_path.read_text("utf-8")) if papers_path.exists() else []
    return f"Output folder: {out} ({len(_state['papers'])} papers already recorded)."


@server.tool()
async def search_arxiv(
    query: Annotated[str, Field(description='arXiv syntax: ti:, abs:, au:, cat: with AND/OR/ANDNOT and quoted '
                                            'phrases, e.g. abs:"lattice" AND abs:"graph neural network"')],
    max_results: Annotated[int, Field(ge=1, le=30)] = 10,
) -> str:
    """Search arXiv. Returns title, authors, year, arxiv_id, doi, pdf_url and a clipped abstract."""
    return _json(await asyncio.to_thread(T.search_arxiv, query, max_results))


@server.tool()
async def search_openalex(
    query: str,
    limit: Annotated[int, Field(ge=1, le=50)] = 10,
    from_year: Annotated[int | None, Field(description="Only papers published in or after this year")] = None,
) -> str:
    """Search OpenAlex (journals, conferences, preprints) by keywords. Returns citation counts, DOIs,
    venues, open-access pdf_url and abstracts. On HTTP 429 the daily quota is used up: use search_arxiv."""
    return _json(await asyncio.to_thread(T.search_openalex, query, limit, from_year))


@server.tool()
async def get_paper_links(
    paper_id: Annotated[str, Field(description="OpenAlex id (e.g. W4404421437) or a DOI")],
    direction: Literal["citations", "references"],
    limit: Annotated[int, Field(ge=1, le=100)] = 30,
) -> str:
    """Snowball from a key paper: 'citations' lists newer papers citing it, 'references' what it cites.
    Sorted by citation count."""
    return _json(await asyncio.to_thread(T.oa_links, paper_id, direction, limit))


@server.tool()
async def read_pdf(
    url: Annotated[str, Field(description="pdf_url from a search result")],
    first_page: Annotated[int, Field(ge=1)] = 1,
    last_page: Annotated[int, Field(ge=1)] = 3,
) -> str:
    """Read an open-access PDF page by page; each page's text follows a '=== PDF page N ===' marker,
    at most 8 pages per call. Start with pages 1-3, then jump to methods/results. Page numbers are PDF
    pages of this version, which can differ from the journal's printed numbers. Publishers often block
    downloads; then look for an arXiv or author copy, otherwise cite the abstract only."""
    return await asyncio.to_thread(T.pdf_pages, url, first_page, last_page)


class Evidence(BaseModel):
    claim: str
    location: str = Field(description="e.g. 'Table 2', 'Fig. 4', 'Sec. 3.1', 'abstract'")
    page: int | None = Field(None, description="PDF page number from read_pdf; omit if abstract only")
    quote: str | None = Field(None, description="short verbatim snippet, at most 25 words")


@server.tool()
async def record_paper(
    title: str,
    authors: str,
    year: int,
    identifier: Annotated[str, Field(description="DOI (preferred) or arXiv id, exactly as a search tool returned it")],
    category: Literal[tuple(T.CATEGORIES)],  # type: ignore[valid-type]
    relevance: Annotated[int, Field(ge=1, le=5, description="5 = directly applicable to this project")],
    summary: Annotated[str, Field(description="2-3 sentences: what the paper does")],
    relevance_to_project: Annotated[str, Field(description="what we should take from it")],
    fulltext_source: Annotated[str, Field(description="PDF url read with read_pdf, or 'abstract only'")],
    evidence: Annotated[list[Evidence], Field(description="pinpointed support for each claim you will make")],
    venue: str | None = None,
    url: str | None = None,
    methods: Annotated[str | None, Field(description="model / representation / training details")] = None,
    data: Annotated[str | None, Field(description="dataset, size, how labels were generated")] = None,
    key_findings: Annotated[str | None, Field(description="quantitative results, with units")] = None,
) -> str:
    """Add one relevant paper to papers.json. Only record papers a search tool actually returned.
    Recording the same identifier again replaces the earlier entry."""
    out = _out_dir()
    entry = {k: v for k, v in locals().items() if k not in ("out", "evidence") and v is not None}
    entry["evidence"] = [e.model_dump(exclude_none=True) for e in evidence]
    key = identifier.strip().lower()
    papers = [p for p in _state["papers"] if p["identifier"].strip().lower() != key] + [entry]
    _state["papers"] = papers
    (out / "papers.json").write_text(json.dumps(papers, indent=2, ensure_ascii=False), "utf-8")
    return f"Recorded ({len(papers)} papers so far)."


@server.tool()
async def write_review(markdown: str) -> str:
    """Write the final review to lit_review.md. Cite as [Author Year, p. N] and end with a reference
    list built from the recorded papers."""
    path = _out_dir() / "lit_review.md"
    path.write_text(markdown, "utf-8")
    return f"Wrote {path} ({len(markdown)} chars). Bibliography has {len(_state['papers'])} papers."


if __name__ == "__main__":
    server.run("stdio")
