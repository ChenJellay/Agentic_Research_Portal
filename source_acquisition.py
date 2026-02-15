"""
Automated source acquisition for the RAG corpus.

Searches arXiv via the public Atom API for papers matching the research domain,
downloads PDFs to ``data/raw/``, and auto-populates ``manifest.json``.

Usage (standalone)::

    python source_acquisition.py --search "AI software development lifecycle" --max 10
    python source_acquisition.py --search "LLM code generation" --max 5

The module can also be called from ``rag_pipeline.py acquire``.

arXiv API
---------
Uses the API at https://export.arxiv.org/api/query (see
https://info.arxiv.org/help/api/basics.html). Parameters: search_query, start,
max_results, sortBy, sortOrder. Returns Atom 1.0 XML. Respects arXiv's 3s delay
recommendation for repeated calls.

Selection strategy
------------------
* Keyword search across all arXiv (no category filter), sorted by relevance.
* Duplicates (by title similarity) are skipped automatically.
"""

import argparse
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from config import get_path_config
from logger_config import setup_logger
from manifest import add_source, load_manifest

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# arXiv acquisition
# ---------------------------------------------------------------------------

# User-Agent for arXiv requests (use a real contact for production; helps avoid rate limits)
_ARXIV_USER_AGENT = "RAG-Corpus-Acquisition/1.0 (https://github.com/your-org/your-repo)"

# Rate-limit mitigation: arXiv allows 1 req/3s; use longer delays for headroom
_DELAY_BETWEEN_SEARCHES = 6  # seconds between search API calls
_DELAY_BETWEEN_DOWNLOADS = 3  # seconds between PDF downloads
_INITIAL_DELAY = 2  # seconds before first request (avoids back-to-back runs)


def _wait_on_rate_limit(resp: requests.Response, attempt: int) -> int:
    """Compute wait time: Retry-After header, or exponential backoff for 429/503."""
    retry_after = resp.headers.get("Retry-After")
    if retry_after:
        try:
            return int(retry_after)
        except ValueError:
            pass
    return 15 * (2**attempt)  # 15s, 30s, 60s


def search_arxiv(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Search arXiv via the public Atom API and return paper metadata.

    Implements the API from https://info.arxiv.org/help/api/basics.html:
    HTTP GET to export.arxiv.org/api/query with search_query, start, max_results,
    sortBy, sortOrder. Parses Atom 1.0 response with xml.etree (no feedparser).
    """
    base_url = "https://export.arxiv.org/api/query"
    # Match arXiv web UI: all:query (no category filter) — cat:cs.SE was too restrictive (0 hits)
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    headers = {"User-Agent": _ARXIV_USER_AGENT}
    logger.info(f"Searching arXiv for: {query}  (max {max_results})")

    for attempt in range(4):
        try:
            resp = requests.get(base_url, params=params, headers=headers, timeout=90)
            if resp.status_code in (429, 503):
                if attempt < 3:
                    wait = _wait_on_rate_limit(resp, attempt)
                    logger.warning(
                        f"arXiv {resp.status_code}, retrying in {wait}s (attempt {attempt + 1}/4)"
                    )
                    time.sleep(wait)
                else:
                    resp.raise_for_status()
            else:
                resp.raise_for_status()
                break
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt < 3:
                wait = 20 * (attempt + 1)
                logger.warning(
                    f"arXiv request timed out, retrying in {wait}s (attempt {attempt + 1}/4)"
                )
                time.sleep(wait)
            else:
                raise

    # Minimal XML parse (avoid lxml dependency)
    entries = _parse_arxiv_atom(resp.text)
    logger.info(f"arXiv returned {len(entries)} result(s)")
    return entries


def _parse_arxiv_atom(xml_text: str) -> List[Dict[str, Any]]:
    """Very lightweight parser for the arXiv Atom feed."""
    import xml.etree.ElementTree as ET

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_text)
    results: List[Dict[str, Any]] = []

    for entry in root.findall("atom:entry", ns):
        arxiv_id_raw = entry.findtext("atom:id", "", ns)
        if "/api/errors" in arxiv_id_raw:
            continue  # skip error entries (API returns errors as Atom entries)
        arxiv_id = arxiv_id_raw.split("/abs/")[-1] if "/abs/" in arxiv_id_raw else arxiv_id_raw
        # Remove version suffix for filename
        arxiv_id_clean = re.sub(r"v\d+$", "", arxiv_id)

        title = (entry.findtext("atom:title", "", ns) or "").strip().replace("\n", " ")
        authors = [
            a.findtext("atom:name", "", ns)
            for a in entry.findall("atom:author", ns)
        ]
        published = entry.findtext("atom:published", "", ns)[:4]  # year
        summary = (entry.findtext("atom:summary", "", ns) or "").strip()[:200]

        pdf_link = ""
        for link in entry.findall("atom:link", ns):
            if link.get("title") == "pdf":
                pdf_link = link.get("href", "")

        results.append({
            "arxiv_id": arxiv_id_clean,
            "title": title,
            "authors": authors,
            "year": int(published) if published.isdigit() else 2024,
            "pdf_url": pdf_link or f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf",
            "link": f"https://arxiv.org/abs/{arxiv_id_clean}",
            "summary": summary,
        })

    return results


def download_arxiv_pdf(arxiv_id: str, pdf_url: str, dest_dir: Path) -> Optional[Path]:
    """Download a single arXiv PDF. Returns path on success, None on failure."""
    safe_name = arxiv_id.replace("/", "_").replace(".", "_")
    dest = dest_dir / f"arxiv_{safe_name}.pdf"
    if dest.exists():
        logger.info(f"Already downloaded: {dest.name}")
        return dest
    headers = {"User-Agent": _ARXIV_USER_AGENT}
    for attempt in range(4):
        try:
            logger.info(f"Downloading {pdf_url} → {dest.name}")
            resp = requests.get(pdf_url, headers=headers, timeout=90)
            if resp.status_code in (429, 503):
                if attempt < 3:
                    wait = _wait_on_rate_limit(resp, attempt)
                    logger.warning(
                        f"Download {resp.status_code}, retrying in {wait}s (attempt {attempt + 1}/4)"
                    )
                    time.sleep(wait)
                else:
                    resp.raise_for_status()
            else:
                resp.raise_for_status()
                dest.write_bytes(resp.content)
                return dest
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt < 3:
                wait = 15 * (attempt + 1)
                logger.warning(
                    f"Download timed out, retrying in {wait}s (attempt {attempt + 1}/4)"
                )
                time.sleep(wait)
            else:
                logger.error(f"Failed to download {pdf_url}: {e}")
                return None
        except Exception as e:
            logger.error(f"Failed to download {pdf_url}: {e}")
            return None
    return None


# ---------------------------------------------------------------------------
# Validation filters
# ---------------------------------------------------------------------------

def filter_by_recency(
    papers: List[Dict[str, Any]], min_year: int
) -> List[Dict[str, Any]]:
    """Keep only papers published in *min_year* or later."""
    filtered = [p for p in papers if p.get("year", 0) >= min_year]
    removed = len(papers) - len(filtered)
    if removed:
        logger.info(f"Recency filter: removed {removed} paper(s) older than {min_year}")
    return filtered


# ---------------------------------------------------------------------------
# Orchestrator: search + download + manifest update
# ---------------------------------------------------------------------------

def _title_similarity(t1: str, t2: str) -> float:
    """Jaccard similarity on lowered word sets — cheap duplicate check."""
    s1 = set(t1.lower().split())
    s2 = set(t2.lower().split())
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def acquire_sources(
    search_queries: List[str],
    max_per_query: int = 10,
    manifest_path: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
    min_year: Optional[int] = None,
    require_peer_reviewed: bool = False,
) -> int:
    """
    Run the full acquisition pipeline:
      1. Search arXiv for each query.
      2. Apply optional recency filter.
      3. De-duplicate against existing manifest titles.
      4. Download PDFs.
      5. Add entries to manifest.

    Args:
        search_queries: One or more keyword queries.
        max_per_query: Max results per query (arXiv API limit 2000/call).
        manifest_path: Override path to ``manifest.json``.
        raw_dir: Override path to the raw PDF directory.
        min_year: If set, discard papers published before this year.
        require_peer_reviewed: Ignored (arXiv-only; all papers are preprints).

    Returns the number of newly acquired sources.
    """
    paths = get_path_config()
    manifest_path = manifest_path or paths.manifest_path
    raw_dir = raw_dir or paths.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    existing = load_manifest(manifest_path)
    existing_titles = [s["title"] for s in existing]

    time.sleep(_INITIAL_DELAY)  # avoid back-to-back runs triggering rate limit
    candidates: List[Dict[str, Any]] = []

    for q in search_queries:
        try:
            candidates.extend(search_arxiv(q, max_results=max_per_query))
        except Exception as e:
            logger.error(f"arXiv search failed for '{q}': {e}")
        time.sleep(_DELAY_BETWEEN_SEARCHES)

    if min_year is not None:
        candidates = filter_by_recency(candidates, min_year)

    added = 0
    for paper in candidates:
        # Skip duplicates
        if any(_title_similarity(paper["title"], et) > 0.7 for et in existing_titles):
            logger.debug(f"Skipping duplicate: {paper['title'][:60]}")
            continue

        # Download (arXiv only)
        pdf_path = download_arxiv_pdf(paper["arxiv_id"], paper["pdf_url"], raw_dir)

        if pdf_path is None:
            continue

        # Build manifest entry
        source_id = pdf_path.stem  # e.g. arxiv_2409_18048
        entry = {
            "source_id": source_id,
            "title": paper["title"],
            "authors": paper["authors"],
            "year": paper["year"],
            "type": "preprint",
            "venue": "arXiv",
            "link": paper["link"],
            "doi": paper.get("doi"),
            "relevance_note": paper.get("summary", "")[:200],
            "filename": pdf_path.name,
            "acquisition_method": "automated",
        }

        try:
            add_source(manifest_path, entry, overwrite=False)
            existing_titles.append(paper["title"])
            added += 1
            logger.info(f"Added source: {source_id}")
        except Exception as e:
            logger.error(f"Failed to add {source_id}: {e}")

        time.sleep(_DELAY_BETWEEN_DOWNLOADS)

    logger.info(f"Acquisition complete. {added} new source(s) added.")
    return added


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire research sources for RAG corpus")
    parser.add_argument(
        "--search", type=str, nargs="+", required=True,
        help="Search query strings (one or more)",
    )
    parser.add_argument(
        "--max", type=int, default=10,
        help="Max results per query per API (default: 10)",
    )
    args = parser.parse_args()
    acquire_sources(args.search, max_per_query=args.max)


if __name__ == "__main__":
    main()
