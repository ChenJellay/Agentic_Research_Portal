"""
API integration tests for Phase 3 PRP endpoints.

Mocks RAG engine and MLX to avoid loading models. Tests routing, validation, and response shape.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def mock_rag_engine():
    """Mock RAGQueryEngine that returns canned answer and chunks."""
    engine = MagicMock()
    engine.query.return_value = (
        "AI affects code review by [arxiv_2409_18048, arxiv_2409_18048_chunk_0001] automating tasks.",
        [
            {
                "chunk_id": "arxiv_2409_18048_chunk_0001",
                "source_id": "arxiv_2409_18048",
                "section": "Intro",
                "text": "AI tools automate code review.",
            },
        ],
    )
    engine.manifest = []
    engine.retriever.retrieve.return_value = [
        {"chunk_id": "c1", "source_id": "s1", "text": "chunk text"},
    ]
    return engine


@pytest.fixture
def client_with_mock_engine(mock_rag_engine, temp_project_root, mock_path_config):
    """TestClient with mocked RAG engine and path config."""
    with patch("app.routes.query._get_engine", return_value=mock_rag_engine), \
         patch("app.routes.query.get_path_config", return_value=mock_path_config), \
         patch("thread_store.get_path_config", return_value=mock_path_config), \
         patch("app.routes.evaluation._get_engine", return_value=mock_rag_engine), \
         patch("app.routes.evaluation.get_path_config", return_value=mock_path_config):
        yield TestClient(app)


@pytest.fixture
def client_threads_only(temp_project_root, mock_path_config):
    """Client for threads-only tests (no query mock)."""
    with patch("thread_store.get_path_config", return_value=mock_path_config):
        yield TestClient(app)


def test_root_endpoint():
    """Root returns API info."""
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert "message" in data
    assert "docs" in data


def test_query_empty_question(client_with_mock_engine):
    """POST /api/query rejects empty question."""
    r = client_with_mock_engine.post("/api/query", json={"question": "   "})
    assert r.status_code == 400


def test_query_success(client_with_mock_engine):
    """POST /api/query returns answer, chunks, citations, thread_id."""
    r = client_with_mock_engine.post(
        "/api/query",
        json={"question": "How does AI affect code review?"},
    )
    assert r.status_code == 200
    data = r.json()
    assert "thread_id" in data
    assert "answer" in data
    assert "retrieved_chunks" in data
    assert "citations" in data
    assert "source_metadata" in data
    assert len(data["retrieved_chunks"]) >= 1


def test_search_empty_query(client_with_mock_engine):
    """GET /api/search rejects empty query."""
    r = client_with_mock_engine.get("/api/search", params={"query": "   "})
    assert r.status_code == 400


def test_search_success(client_with_mock_engine):
    """GET /api/search returns chunks without generation."""
    r = client_with_mock_engine.get(
        "/api/search",
        params={"query": "AI code review", "top_k": 3},
    )
    assert r.status_code == 200
    data = r.json()
    assert "chunks" in data


def test_threads_list_empty(client_threads_only):
    """GET /api/threads returns empty list when no threads."""
    r = client_threads_only.get("/api/threads")
    assert r.status_code == 200
    assert r.json()["threads"] == []


def test_threads_list_after_query(client_with_mock_engine):
    """After a query, thread appears in list."""
    client_with_mock_engine.post(
        "/api/query",
        json={"question": "How does AI affect code review?"},
    )
    r = client_with_mock_engine.get("/api/threads")
    assert r.status_code == 200
    threads = r.json()["threads"]
    assert len(threads) >= 1
    assert "thread_id" in threads[0]
    assert "query" in threads[0]


def test_thread_get_not_found(client_threads_only):
    """GET /api/threads/{id} returns 404 for unknown thread."""
    r = client_threads_only.get("/api/threads/nonexistent-id")
    assert r.status_code == 404


def test_thread_get_success(client_with_mock_engine):
    """GET /api/threads/{id} returns full thread after query."""
    qr = client_with_mock_engine.post(
        "/api/query",
        json={"question": "How does AI affect code review?"},
    )
    thread_id = qr.json()["thread_id"]

    r = client_with_mock_engine.get(f"/api/threads/{thread_id}")
    assert r.status_code == 200
    data = r.json()
    assert data["thread_id"] == thread_id
    assert data["query"] == "How does AI affect code review?"
    assert "answer" in data
    assert "retrieved_chunks" in data
