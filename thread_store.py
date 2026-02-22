"""
Research thread store — Phase 3.

File-based persistence for research threads: query + retrieved evidence + answer.
Each thread is saved as data/threads/{thread_id}.json.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import get_path_config


def _threads_dir() -> Path:
    """Return threads directory, ensuring it exists."""
    d = get_path_config().threads_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_thread(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    answer: str,
    citations: List[Dict[str, str]],
    source_metadata: Dict[str, Dict[str, Any]],
    suggested_queries: Optional[List[str]] = None,
    thread_id: Optional[str] = None,
) -> str:
    """
    Save a research thread to disk.

    Args:
        query: The user's question.
        retrieved_chunks: Chunks returned by the retriever.
        answer: Model-generated answer.
        citations: Extracted citations [{"source_id", "chunk_id"}].
        source_metadata: Manifest metadata for cited sources.
        suggested_queries: Optional suggested next retrieval steps.
        thread_id: If provided, overwrite existing thread; else create new.

    Returns:
        The thread_id (new or existing).
    """
    tid = thread_id or str(uuid.uuid4())
    path = _threads_dir() / f"{tid}.json"

    record: Dict[str, Any] = {
        "thread_id": tid,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "retrieved_chunks": retrieved_chunks,
        "answer": answer,
        "citations": citations,
        "source_metadata": source_metadata,
    }
    if suggested_queries:
        record["suggested_queries"] = suggested_queries

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False)

    return tid


def load_thread(thread_id: str) -> Optional[Dict[str, Any]]:
    """
    Load a thread by ID.

    Returns:
        Thread dict or None if not found.
    """
    path = _threads_dir() / f"{thread_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def list_threads() -> List[Dict[str, Any]]:
    """
    List all threads, sorted by created_at descending.

    Returns:
        List of thread summaries: {thread_id, created_at, query}.
    """
    d = _threads_dir()
    if not d.exists():
        return []

    summaries: List[Dict[str, Any]] = []
    for path in d.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            summaries.append({
                "thread_id": data.get("thread_id", path.stem),
                "created_at": data.get("created_at", ""),
                "query": data.get("query", "")[:100],
            })
        except (json.JSONDecodeError, IOError):
            continue

    summaries.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return summaries
