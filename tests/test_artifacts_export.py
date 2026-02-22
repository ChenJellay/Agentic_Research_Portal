"""
Tests for artifact generation and export endpoints.

Uses mocked artifact_generator to avoid LLM calls. Verifies routing and response structure.
"""

from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def mock_artifact_responses():
    """Canned responses from artifact generator."""
    return {
        "evidence_table": {
            "rows": [
                {
                    "claim": "AI automates code review",
                    "evidence_snippet": "AI tools automate...",
                    "source_id": "s1",
                    "chunk_id": "c1",
                    "confidence": 0.9,
                    "notes": "",
                },
            ],
            "markdown": "| Claim | Evidence | ... |",
        },
        "annotated_bib": {
            "entries": [
                {
                    "source_id": "s1",
                    "title": "Paper",
                    "claim": "Main claim",
                    "method": "Study",
                    "limitations": "Small sample",
                    "why_it_matters": "Relevant",
                },
            ],
            "markdown": "### s1: Paper\n...",
        },
        "synthesis_memo": {
            "content": "# Synthesis Memo\n\nFormal memo with [s1, c1] citations.",
            "markdown": "# Synthesis Memo\n\nFormal memo with [s1, c1] citations.",
        },
    }


@pytest.fixture
def client_with_thread_and_artifacts(temp_project_root, mock_path_config, sample_thread, mock_artifact_responses):
    """Client with a saved thread and mocked artifact generator."""
    import json
    threads_dir = temp_project_root / "data" / "threads"
    threads_dir.mkdir(parents=True, exist_ok=True)
    thread_path = threads_dir / f"{sample_thread['thread_id']}.json"
    thread_path.write_text(json.dumps(sample_thread), encoding="utf-8")

    def mock_evidence_table(thread):
        return mock_artifact_responses["evidence_table"]

    def mock_annotated_bib(thread):
        return mock_artifact_responses["annotated_bib"]

    def mock_synthesis_memo(thread):
        return mock_artifact_responses["synthesis_memo"]

    with patch("thread_store.get_path_config", return_value=mock_path_config), \
         patch("app.routes.artifacts.load_thread", return_value=sample_thread), \
         patch("app.routes.artifacts.generate_evidence_table", side_effect=mock_evidence_table), \
         patch("app.routes.artifacts.generate_annotated_bib", side_effect=mock_annotated_bib), \
         patch("app.routes.artifacts.generate_synthesis_memo", side_effect=mock_synthesis_memo), \
         patch("app.routes.export.load_thread", return_value=sample_thread), \
         patch("app.routes.export.generate_evidence_table", side_effect=mock_evidence_table), \
         patch("app.routes.export.generate_annotated_bib", side_effect=mock_annotated_bib), \
         patch("app.routes.export.generate_synthesis_memo", side_effect=mock_synthesis_memo):
        yield TestClient(app)


def test_evidence_table_404(client_with_thread_and_artifacts):
    """Evidence table returns 404 for unknown thread."""
    with patch("app.routes.artifacts.load_thread", return_value=None):
        r = client_with_thread_and_artifacts.post(
            "/api/artifacts/evidence-table",
            json={"thread_id": "unknown"},
        )
        assert r.status_code == 404


def test_evidence_table_success(client_with_thread_and_artifacts, sample_thread):
    """POST /api/artifacts/evidence-table returns rows and markdown."""
    r = client_with_thread_and_artifacts.post(
        "/api/artifacts/evidence-table",
        json={"thread_id": sample_thread["thread_id"]},
    )
    assert r.status_code == 200
    data = r.json()
    assert "rows" in data
    assert "markdown" in data
    assert len(data["rows"]) >= 1


def test_annotated_bib_success(client_with_thread_and_artifacts, sample_thread):
    """POST /api/artifacts/annotated-bib returns entries and markdown."""
    r = client_with_thread_and_artifacts.post(
        "/api/artifacts/annotated-bib",
        json={"thread_id": sample_thread["thread_id"]},
    )
    assert r.status_code == 200
    data = r.json()
    assert "entries" in data
    assert "markdown" in data


def test_synthesis_memo_success(client_with_thread_and_artifacts, sample_thread):
    """POST /api/artifacts/synthesis-memo returns content."""
    r = client_with_thread_and_artifacts.post(
        "/api/artifacts/synthesis-memo",
        json={"thread_id": sample_thread["thread_id"]},
    )
    assert r.status_code == 200
    data = r.json()
    assert "content" in data or "markdown" in data


def test_export_md_evidence_table(client_with_thread_and_artifacts, sample_thread):
    """Export evidence table as Markdown."""
    r = client_with_thread_and_artifacts.get(
        f"/api/export/md",
        params={"artifact_type": "evidence-table", "thread_id": sample_thread["thread_id"]},
    )
    assert r.status_code == 200
    assert "text/markdown" in r.headers.get("content-type", "")
    assert "Content-Disposition" in r.headers
    assert "attachment" in r.headers["Content-Disposition"]


def test_export_csv_evidence_table(client_with_thread_and_artifacts, sample_thread):
    """Export evidence table as CSV."""
    r = client_with_thread_and_artifacts.get(
        f"/api/export/csv",
        params={"artifact_type": "evidence-table", "thread_id": sample_thread["thread_id"]},
    )
    assert r.status_code == 200
    assert "text/csv" in r.headers.get("content-type", "")
    assert "Claim" in r.text or "claim" in r.text.lower()


def test_export_synthesis_memo_csv_rejected(client_with_thread_and_artifacts, sample_thread):
    """Synthesis memo cannot be exported as CSV."""
    r = client_with_thread_and_artifacts.get(
        f"/api/export/csv",
        params={"artifact_type": "synthesis-memo", "thread_id": sample_thread["thread_id"]},
    )
    assert r.status_code == 400


def test_export_unknown_format(client_with_thread_and_artifacts, sample_thread):
    """Unknown format returns 400."""
    r = client_with_thread_and_artifacts.get(
        f"/api/export/xyz",
        params={"artifact_type": "evidence-table", "thread_id": sample_thread["thread_id"]},
    )
    assert r.status_code == 400
