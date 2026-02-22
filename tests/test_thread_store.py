"""
Unit tests for thread_store — Phase 3 file-based thread persistence.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from thread_store import save_thread, load_thread, list_threads


@pytest.fixture
def threads_dir(tmp_path):
    """Temp directory for thread files."""
    d = tmp_path / "threads"
    d.mkdir(parents=True)
    return d


def test_save_and_load_thread(threads_dir):
    """Save a thread and load it back."""
    with patch("thread_store.get_path_config") as mock_cfg:
        mock_cfg.return_value.threads_dir = threads_dir

        tid = save_thread(
            query="How does AI affect testing?",
            retrieved_chunks=[{"chunk_id": "c1", "source_id": "s1", "text": "AI helps."}],
            answer="AI helps with testing [s1, c1].",
            citations=[{"source_id": "s1", "chunk_id": "c1"}],
            source_metadata={"s1": {"title": "Paper"}},
        )

        assert tid is not None
        assert len(tid) == 36  # UUID format

        loaded = load_thread(tid)
        assert loaded is not None
        assert loaded["query"] == "How does AI affect testing?"
        assert loaded["answer"] == "AI helps with testing [s1, c1]."
        assert len(loaded["retrieved_chunks"]) == 1
        assert loaded["retrieved_chunks"][0]["chunk_id"] == "c1"
        assert "created_at" in loaded


def test_save_thread_with_suggested_queries(threads_dir):
    """Thread with suggested_queries is persisted correctly."""
    with patch("thread_store.get_path_config") as mock_cfg:
        mock_cfg.return_value.threads_dir = threads_dir

        tid = save_thread(
            query="Obscure topic",
            retrieved_chunks=[],
            answer="No sufficient evidence found in the corpus. Suggested next steps: [query1], [query2]",
            citations=[],
            source_metadata={},
            suggested_queries=["query1", "query2"],
        )

        loaded = load_thread(tid)
        assert loaded["suggested_queries"] == ["query1", "query2"]


def test_load_nonexistent_thread(threads_dir):
    """Loading non-existent thread returns None."""
    with patch("thread_store.get_path_config") as mock_cfg:
        mock_cfg.return_value.threads_dir = threads_dir

        result = load_thread("nonexistent-uuid-12345")
        assert result is None


def test_list_threads(threads_dir):
    """List threads returns summaries sorted by created_at desc."""
    with patch("thread_store.get_path_config") as mock_cfg:
        mock_cfg.return_value.threads_dir = threads_dir

        # Create two threads manually to control order
        for i, q in enumerate(["First query", "Second query"]):
            path = threads_dir / f"thread-{i}.json"
            path.write_text(
                json.dumps({
                    "thread_id": f"thread-{i}",
                    "created_at": f"2025-02-2{i}Z",
                    "query": q,
                })
            )

        threads = list_threads()
        assert len(threads) == 2
        # Sorted desc by created_at: 2025-02-22 > 2025-02-21
        assert threads[0]["query"] == "Second query"
        assert threads[1]["query"] == "First query"


def test_save_with_explicit_thread_id(threads_dir):
    """Can overwrite with explicit thread_id."""
    with patch("thread_store.get_path_config") as mock_cfg:
        mock_cfg.return_value.threads_dir = threads_dir

        tid = save_thread(
            query="Q1",
            retrieved_chunks=[],
            answer="A1",
            citations=[],
            source_metadata={},
            thread_id="my-custom-id",
        )

        assert tid == "my-custom-id"
        loaded = load_thread("my-custom-id")
        assert loaded["query"] == "Q1"
