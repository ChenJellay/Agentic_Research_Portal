"""
Data-manifest management for the RAG corpus.

Handles CRUD operations on ``manifest.json``, validates required fields,
and provides look-up helpers used by the ingestion and query pipelines.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from logger_config import setup_logger

logger = setup_logger(__name__)

# Required fields for every source entry
REQUIRED_FIELDS = [
    "source_id",
    "title",
    "authors",
    "year",
    "type",
    "venue",
    "link",
    "relevance_note",
    "filename",
]

# Optional but tracked fields
OPTIONAL_FIELDS = ["doi", "acquisition_method"]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    """Load manifest.json and return the list of source entries."""
    if not manifest_path.exists():
        logger.warning(f"Manifest not found at {manifest_path}; returning empty list.")
        return []
    with open(manifest_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "sources" in data:
        return data["sources"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected manifest format in {manifest_path}")


def save_manifest(manifest_path: Path, sources: List[Dict[str, Any]]) -> None:
    """Persist the source list to manifest.json (pretty-printed)."""
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump({"sources": sources}, fh, indent=2, ensure_ascii=False)
    logger.info(f"Manifest saved ({len(sources)} sources) → {manifest_path}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_entry(entry: Dict[str, Any]) -> List[str]:
    """Return a list of validation error strings (empty == valid)."""
    errors: List[str] = []
    for field in REQUIRED_FIELDS:
        if field not in entry or entry[field] is None:
            errors.append(f"Missing required field: {field}")
    if "year" in entry and entry["year"] is not None:
        if not isinstance(entry["year"], int):
            errors.append(f"'year' must be an integer, got {type(entry['year']).__name__}")
    if "authors" in entry and entry["authors"] is not None:
        if not isinstance(entry["authors"], list):
            errors.append(f"'authors' must be a list, got {type(entry['authors']).__name__}")
    return errors


def validate_manifest(sources: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Validate every entry; return {source_id: [errors]} for invalid ones."""
    problems: Dict[str, List[str]] = {}
    seen_ids: set = set()
    for idx, entry in enumerate(sources):
        sid = entry.get("source_id", f"<entry_{idx}>")
        errs = validate_entry(entry)
        if sid in seen_ids:
            errs.append(f"Duplicate source_id: {sid}")
        seen_ids.add(sid)
        if errs:
            problems[sid] = errs
    return problems


# ---------------------------------------------------------------------------
# Look-up helpers
# ---------------------------------------------------------------------------

def get_source_by_id(
    sources: List[Dict[str, Any]], source_id: str
) -> Optional[Dict[str, Any]]:
    """Find a single source entry by its source_id."""
    for s in sources:
        if s.get("source_id") == source_id:
            return s
    return None


def get_all_source_ids(sources: List[Dict[str, Any]]) -> List[str]:
    return [s["source_id"] for s in sources]


def get_filename_to_id_map(sources: List[Dict[str, Any]]) -> Dict[str, str]:
    """Return {filename: source_id} mapping."""
    return {s["filename"]: s["source_id"] for s in sources}


# ---------------------------------------------------------------------------
# Mutation helpers
# ---------------------------------------------------------------------------

def add_source(
    manifest_path: Path, entry: Dict[str, Any], overwrite: bool = False
) -> None:
    """Add or update a source entry in the manifest file."""
    sources = load_manifest(manifest_path)
    errs = validate_entry(entry)
    if errs:
        raise ValueError(f"Invalid entry: {'; '.join(errs)}")
    existing = get_source_by_id(sources, entry["source_id"])
    if existing and not overwrite:
        logger.info(f"Source {entry['source_id']} already in manifest; skipping.")
        return
    if existing:
        sources = [s for s in sources if s["source_id"] != entry["source_id"]]
    sources.append(entry)
    save_manifest(manifest_path, sources)


def remove_source(manifest_path: Path, source_id: str) -> bool:
    """Remove a source entry by ID.  Returns True if removed."""
    sources = load_manifest(manifest_path)
    new = [s for s in sources if s["source_id"] != source_id]
    if len(new) == len(sources):
        return False
    save_manifest(manifest_path, new)
    return True
