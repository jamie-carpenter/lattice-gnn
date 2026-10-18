"""Custom tools for the literature agent, served to Claude as an in-process MCP server.

  search_arxiv            arXiv API (preprints; good for recent ML work)
  search_openalex         OpenAlex (journals + preprints, with citation counts)
  get_paper_links         papers that cite / are cited by a given paper (snowballing)
  read_pdf                full text of an open-access PDF, split by page (for page citations)
  record_paper            append a structured entry to papers.json
  write_review            write the final lit_review.md

The search tools only use free public APIs and the standard library. Without a key, OpenAlex
allows roughly 100 requests/day shared by everyone on your IP; set OPENALEX_API_KEY to use a free
personal key (about 10x that), from https://openalex.org/settings/api.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path
from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool
from pypdf import PdfReader

USER_AGENT = "lattice-gnn-literature-agent/0.1"
OPENALEX = "https://api.openalex.org"
OA_FIELDS = "id,doi,title,publication_year,authorships,primary_location,cited_by_count,abstract_inverted_index,best_oa_location"
ARXIV_NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

CATEGORIES = [
    "dataset",                 # lattice / metamaterial datasets with elastic properties
    "surrogate_model",         # ML models predicting stiffness / elastic tensors
    "gnn_architecture",        # graph networks for periodic structures / crystals
    "symmetry_equivariance",   # tensor-valued outputs, equivariance, physical constraints
    "homogenization",          # FE / analytical homogenization used to make labels
    "defects_manufacturing",   # as-built geometry, imperfections, knockdown factors
    "inverse_design",          # generative / optimization over lattice designs
    "other",
]


# --------------------------------------------------------------------------- HTTP

_rate_lock = threading.Lock()
_last_call: dict[str, float] = {}
_MIN_SPACING = {"arxiv": 3.0, "openalex": 0.2, "pdf": 1.0}  # seconds between calls, per the APIs' guidance


MAX_PDF_BYTES = 40 * 1024 * 1024


def _http_get(url: str, api: str, headers: dict[str, str] | None = None, retries: int = 4) -> str:
    return _http_fetch(url, api, headers, retries).decode("utf-8")


def _http_fetch(url: str, api: str, headers: dict[str, str] | None = None, retries: int = 4) -> bytes:
    """GET with per-API spacing and exponential backoff on 429 / 5xx."""
    for attempt in range(retries):
        with _rate_lock:
            wait = _last_call.get(api, 0.0) + _MIN_SPACING[api] - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            _last_call[api] = time.monotonic()
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read(MAX_PDF_BYTES + 1)
                if len(body) > MAX_PDF_BYTES:
                    raise ValueError("response larger than 40 MB")
                return body
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2.0 * 2**attempt)
                continue
            raise
    raise RuntimeError("unreachable")


def _openalex(path: str, params: dict[str, Any]) -> dict[str, Any]:
    if key := os.environ.get("OPENALEX_API_KEY"):
        params = {**params, "api_key": key}
    return json.loads(_http_get(f"{OPENALEX}{path}?{urllib.parse.urlencode(params)}", "openalex"))


def _authors(names: list[str], keep: int = 4) -> str:
    return ", ".join(names[:keep]) + (" et al." if len(names) > keep else "")


def _clip(text: str | None, n: int = 700) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n] + "…"


def _ok(payload: Any) -> dict[str, Any]:
    text = payload if isinstance(payload, str) else json.dumps(payload, indent=1, ensure_ascii=False)
    return {"content": [{"type": "text", "text": text}]}


def _err(msg: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": msg}], "is_error": True}


# --------------------------------------------------------------------------- searches

def _arxiv_search(query: str, max_results: int) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode({
        "search_query": query, "start": 0, "max_results": max_results,
        "sortBy": "relevance", "sortOrder": "descending",
    })
    root = ET.fromstring(_http_get(f"https://export.arxiv.org/api/query?{params}", "arxiv"))
    out = []
    for e in root.findall("a:entry", ARXIV_NS):
        arxiv_id = e.findtext("a:id", "", ARXIV_NS).rsplit("/abs/", 1)[-1]
        cat = e.find("arxiv:primary_category", ARXIV_NS)
        out.append({
            "arxiv_id": arxiv_id,
            "title": " ".join(e.findtext("a:title", "", ARXIV_NS).split()),
            "authors": _authors([a.findtext("a:name", "", ARXIV_NS) for a in e.findall("a:author", ARXIV_NS)]),
            "year": e.findtext("a:published", "", ARXIV_NS)[:4],
            "category": cat.get("term") if cat is not None else None,
            "doi": e.findtext("arxiv:doi", None, ARXIV_NS),
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
            "abstract": _clip(e.findtext("a:summary", "", ARXIV_NS)),
        })
    return out


def _abstract(inverted: dict[str, list[int]] | None) -> str | None:
    """OpenAlex stores abstracts as {word: [positions]}; rebuild the text."""
    if not inverted:
        return None
    words = sorted((i, w) for w, pos in inverted.items() for i in pos)
    return _clip(" ".join(w for _, w in words))


def _oa_paper(w: dict[str, Any]) -> dict[str, Any]:
    source = (w.get("primary_location") or {}).get("source") or {}
    return {
        "openalex_id": (w.get("id") or "").rsplit("/", 1)[-1],
        "title": w.get("title"),
        "authors": _authors([a["author"]["display_name"] for a in w.get("authorships") or []]),
        "year": w.get("publication_year"),
        "venue": source.get("display_name"),
        "citations": w.get("cited_by_count"),
        "doi": (w.get("doi") or "").removeprefix("https://doi.org/") or None,
        "pdf_url": (w.get("best_oa_location") or {}).get("pdf_url"),
        "abstract": _abstract(w.get("abstract_inverted_index")),
    }


def _oa_search(query: str, limit: int, from_year: int | None) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"search": query, "per-page": limit, "select": OA_FIELDS}
    if from_year:
        params["filter"] = f"from_publication_date:{from_year}-01-01"
    return [_oa_paper(w) for w in _openalex("/works", params).get("results") or []]


def _oa_work_id(paper_id: str) -> str:
    """Accept an OpenAlex id (W123 or URL), a DOI, or doi:/https://doi.org/ forms."""
    pid = paper_id.strip().rsplit("openalex.org/", 1)[-1]
    if pid[:1] in "Ww" and pid[1:].isdigit():
        return pid.upper()
    doi = pid.removeprefix("https://doi.org/").removeprefix("doi:").removeprefix("DOI:")
    return _openalex(f"/works/doi:{urllib.parse.quote(doi, safe='/')}", {"select": "id"})["id"].rsplit("/", 1)[-1]


def _oa_links(paper_id: str, direction: str, limit: int) -> list[dict[str, Any]]:
    wid = _oa_work_id(paper_id)
    if direction == "citations":
        flt = f"cites:{wid}"
    else:
        refs = _openalex(f"/works/{wid}", {"select": "referenced_works"}).get("referenced_works") or []
        if not refs:
            return []
        flt = "openalex_id:" + "|".join(r.rsplit("/", 1)[-1] for r in refs[:100])
    params = {"filter": flt, "sort": "cited_by_count:desc", "per-page": limit, "select": OA_FIELDS}
    return [_oa_paper(w) for w in _openalex("/works", params).get("results") or []]


MAX_PAGES_PER_CALL = 8
_pdf_cache: dict[str, list[str]] = {}


def _pdf_pages(url: str, first: int, last: int) -> str:
    """Text of pages first..last (1-based PDF page numbers), each under a page marker."""
    if not url.startswith(("https://", "http://")):
        raise ValueError("url must be http(s)")
    if url not in _pdf_cache:
        data = _http_fetch(url, "arxiv" if "arxiv.org" in url else "pdf", {"Accept": "application/pdf"})
        if not data.startswith(b"%PDF"):
            raise ValueError("not a PDF (probably a paywall or landing page); use the abstract instead")
        _pdf_cache[url] = [page.extract_text() or "" for page in PdfReader(BytesIO(data)).pages]
    pages = _pdf_cache[url]
    first = max(1, first)
    last = min(len(pages), last, first + MAX_PAGES_PER_CALL - 1)
    parts = [f"[{url}: PDF has {len(pages)} pages; showing {first}-{last}]"]
    for i in range(first, last + 1):
        text = " ".join(pages[i - 1].split())
        parts.append(f"\n=== PDF page {i} ===\n{text[:9000]}")
    return "\n".join(parts)


# --------------------------------------------------------------------------- server

RECORD_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "authors": {"type": "string"},
        "year": {"type": "integer"},
        "venue": {"type": "string"},
        "identifier": {"type": "string", "description": "DOI (preferred) or arXiv id, exactly as returned by a search tool"},
        "url": {"type": "string"},
        "category": {"type": "string", "enum": CATEGORIES},
        "relevance": {"type": "integer", "minimum": 1, "maximum": 5,
                      "description": "5 = directly applicable to the lattice-GNN stiffness project"},
        "summary": {"type": "string", "description": "2-3 sentences: what the paper does"},
        "methods": {"type": "string", "description": "model / representation / training details"},
        "data": {"type": "string", "description": "dataset, size, how labels were generated"},
        "key_findings": {"type": "string", "description": "quantitative results where available"},
        "relevance_to_project": {"type": "string", "description": "what we should take from it"},
        "fulltext_source": {"type": "string",
                            "description": "PDF url read with read_pdf (page numbers refer to this version), "
                                           "or 'abstract only'"},
        "evidence": {
            "type": "array",
            "description": "Pinpointed support for each specific claim you will make about this paper",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "page": {"type": "integer", "description": "PDF page number from read_pdf; omit if abstract only"},
                    "location": {"type": "string", "description": "e.g. 'Table 2', 'Fig. 4', 'Sec. 3.1', 'abstract'"},
                    "quote": {"type": "string", "description": "short verbatim snippet, at most 25 words"},
                },
                "required": ["claim", "location"],
            },
        },
    },
    "required": ["title", "authors", "year", "identifier", "category", "relevance", "summary", "relevance_to_project",
                 "fulltext_source", "evidence"],
}


def build_server(out_dir: Path):
    """Create the MCP server. Output files go to `out_dir`, and nowhere else."""
    out_dir.mkdir(parents=True, exist_ok=True)
    papers_path = out_dir / "papers.json"
    papers: list[dict[str, Any]] = json.loads(papers_path.read_text("utf-8")) if papers_path.exists() else []

    @tool(
        "search_arxiv",
        "Search arXiv. `query` uses arXiv syntax: field prefixes ti:, abs:, au:, cat:, combined with "
        'AND/OR/ANDNOT and quoted phrases, e.g. abs:"lattice" AND abs:"graph neural network". '
        "Returns title, authors, year, arxiv_id, doi and a clipped abstract.",
        {"type": "object",
         "properties": {"query": {"type": "string"},
                        "max_results": {"type": "integer", "minimum": 1, "maximum": 30, "default": 10}},
         "required": ["query"]},
    )
    async def search_arxiv(args):
        try:
            return _ok(await asyncio.to_thread(_arxiv_search, args["query"], args.get("max_results", 10)))
        except Exception as e:
            return _err(f"arXiv search failed: {e}")

    @tool(
        "search_openalex",
        "Search OpenAlex (journals, conferences and preprints) by keywords. Returns citation counts, "
        "DOIs, venues and abstracts. `from_year` keeps only papers published in or after that year.",
        {"type": "object",
         "properties": {"query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                        "from_year": {"type": "integer"}},
         "required": ["query"]},
    )
    async def search_openalex(args):
        try:
            return _ok(await asyncio.to_thread(_oa_search, args["query"], args.get("limit", 10), args.get("from_year")))
        except Exception as e:
            return _err(f"OpenAlex search failed: {e}. If this is HTTP 429, the daily budget is used up; use search_arxiv and WebSearch.")

    @tool(
        "get_paper_links",
        "Snowball from a key paper. `paper_id` is an OpenAlex id (e.g. W4404421437) or a DOI. "
        "direction='citations' lists newer papers citing it; 'references' lists what it cites. "
        "Results are sorted by citation count.",
        {"type": "object",
         "properties": {"paper_id": {"type": "string"},
                        "direction": {"type": "string", "enum": ["citations", "references"]},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 30}},
         "required": ["paper_id", "direction"]},
    )
    async def get_paper_links(args):
        try:
            return _ok(await asyncio.to_thread(_oa_links, args["paper_id"], args["direction"], args.get("limit", 30)))
        except Exception as e:
            return _err(f"Lookup failed for {args['paper_id']}: {e}")

    @tool(
        "read_pdf",
        "Read an open-access PDF (a pdf_url from a search result) page by page. Each page's text "
        f"comes under a '=== PDF page N ===' marker; at most {MAX_PAGES_PER_CALL} pages per call. "
        "Start with pages 1-3, then jump to the methods/results pages you need. Page numbers are "
        "PDF pages of this specific version, which can differ from the journal's printed numbers.",
        {"type": "object",
         "properties": {"url": {"type": "string"},
                        "first_page": {"type": "integer", "minimum": 1, "default": 1},
                        "last_page": {"type": "integer", "minimum": 1, "default": 3}},
         "required": ["url"]},
    )
    async def read_pdf(args):
        try:
            return _ok(await asyncio.to_thread(_pdf_pages, args["url"], args.get("first_page", 1),
                                               args.get("last_page", 3)))
        except Exception as e:
            return _err(f"Could not read PDF {args['url']}: {e}. Publishers often block automated "
                        "downloads; look for an arXiv or author copy, otherwise cite the abstract only.")

    @tool(
        "record_paper",
        "Add one relevant paper to the bibliography (papers.json). Only record papers that a search "
        "tool actually returned. Recording the same identifier again replaces the earlier entry.",
        RECORD_SCHEMA,
    )
    async def record_paper(args):
        key = args["identifier"].strip().lower()
        papers[:] = [p for p in papers if p["identifier"].strip().lower() != key]
        papers.append(args)
        papers_path.write_text(json.dumps(papers, indent=2, ensure_ascii=False), "utf-8")
        return _ok(f"Recorded ({len(papers)} papers so far).")

    @tool(
        "write_review",
        "Write the final literature review (Markdown) to lit_review.md. Cite papers as "
        "[Author Year] and end with a reference list built from the recorded papers.",
        {"type": "object", "properties": {"markdown": {"type": "string"}}, "required": ["markdown"]},
    )
    async def write_review(args):
        path = out_dir / "lit_review.md"
        path.write_text(args["markdown"], "utf-8")
        return _ok(f"Wrote {path} ({len(args['markdown'])} chars). Bibliography has {len(papers)} papers.")

    return create_sdk_mcp_server(
        name="literature",
        version="0.1.0",
        tools=[search_arxiv, search_openalex, get_paper_links, read_pdf, record_paper, write_review],
    )


TOOL_NAMES = ["search_arxiv", "search_openalex", "get_paper_links", "read_pdf", "record_paper", "write_review"]
