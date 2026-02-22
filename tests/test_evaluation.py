"""
Tests for evaluation API endpoints.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client_eval_latest(temp_project_root, mock_path_config):
    """Client for evaluation/latest (no eval results)."""
    with patch("app.routes.evaluation.get_path_config", return_value=mock_path_config):
        yield TestClient(app)


def test_evaluation_latest_empty(client_eval_latest):
    """GET /api/evaluation/latest returns empty when no results."""
    r = client_eval_latest.get("/api/evaluation/latest")
    assert r.status_code == 200
    data = r.json()
    assert data["summary"] is None
    assert data["per_query"] == []


def test_evaluation_latest_with_results(temp_project_root, mock_path_config):
    """GET /api/evaluation/latest returns data when eval JSON exists."""
    results_dir = temp_project_root / "eval" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    eval_file = results_dir / "eval_20250221_120000.json"
    eval_file.write_text(
        json.dumps({
            "summary": {"groundedness": 0.85, "citation_precision": 0.9},
            "per_query": [{"query": "Q1", "groundedness": 1.0}],
        })
    )

    with patch("app.routes.evaluation.get_path_config", return_value=mock_path_config):
        client = TestClient(app)
        r = client.get("/api/evaluation/latest")
        assert r.status_code == 200
        data = r.json()
        assert data["summary"] is not None
        assert data["summary"]["groundedness"] == 0.85
        assert len(data["per_query"]) >= 1
