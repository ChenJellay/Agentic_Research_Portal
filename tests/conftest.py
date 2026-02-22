"""
Pytest fixtures for Phase 3 PRP tests.

Uses temp directories for threads and mocks heavy dependencies (RAG engine, MLX).
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config import PathConfig


@pytest.fixture
def temp_project_root(tmp_path):
    """Create a temporary project root with required structure."""
    (tmp_path / "data" / "threads").mkdir(parents=True)
    (tmp_path / "data" / "index").mkdir(parents=True)
    (tmp_path / "eval" / "results").mkdir(parents=True)
    (tmp_path / "manifest.json").write_text(json.dumps({"sources": []}))
    return tmp_path


@pytest.fixture
def mock_path_config(temp_project_root):
    """Override PathConfig to use temp directory."""
    config = PathConfig(project_root=temp_project_root)
    return config


@pytest.fixture
def sample_thread():
    """Sample thread dict for artifact/export tests."""
    return {
        "thread_id": "test-thread-123",
        "created_at": "2025-02-21T12:00:00Z",
        "query": "How does AI affect code review?",
        "retrieved_chunks": [
            {
                "chunk_id": "arxiv_2409_18048_chunk_0001",
                "source_id": "arxiv_2409_18048",
                "section": "Introduction",
                "text": "AI tools are increasingly used in code review to automate repetitive tasks.",
            },
            {
                "chunk_id": "ijsr_2024_chunk_0002",
                "source_id": "ijsr_2024",
                "section": "Methods",
                "text": "We conducted a study on 50 developers using AI-assisted code review.",
            },
        ],
        "answer": "AI affects code review by [arxiv_2409_18048, arxiv_2409_18048_chunk_0001] automating tasks.",
        "citations": [
            {"source_id": "arxiv_2409_18048", "chunk_id": "arxiv_2409_18048_chunk_0001"},
        ],
        "source_metadata": {
            "arxiv_2409_18048": {
                "source_id": "arxiv_2409_18048",
                "title": "AI in Software Engineering",
                "authors": ["Author A"],
                "year": 2024,
                "venue": "arXiv",
                "link": "https://arxiv.org/abs/2409.18048",
            },
            "ijsr_2024": {
                "source_id": "ijsr_2024",
                "title": "Developer Study",
                "authors": ["Author B"],
                "year": 2024,
                "venue": "IJSRA",
                "link": "https://example.com",
            },
        },
    }
