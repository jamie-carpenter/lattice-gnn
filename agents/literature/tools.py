"""Search and PDF helpers behind the literature agent's tools (see server.py).

  search_arxiv     arXiv API (preprints; good for recent ML work)
  search_openalex  OpenAlex (journals + preprints, with citation counts)
  oa_links         papers that cite / are cited by a given paper (snowballing)
  pdf_pages        full text of an open-access PDF, split by page (for page citations)

Only free public APIs. Without a key, OpenAlex allows roughly 100 requests/day shared by
everyone on your IP; set OPENALEX_API_KEY to use a free personal key (about 10x that),
from https://openalex.org/settings/api.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Any

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


# --------------------------------------------------------------------------- searches

def search_arxiv(query: str, max_results: int) -> list[dict[str, Any]]:
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


def search_openalex(query: str, limit: int, from_year: int | None) -> list[dict[str, Any]]:
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


def oa_links(paper_id: str, direction: str, limit: int) -> list[dict[str, Any]]:
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


def pdf_pages(url: str, first: int, last: int) -> str:
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
