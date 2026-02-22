"""
Tests for evaluator citation exclusions (Modules 1-4).

Covers:
- _is_in_suggested_next_steps: citations in "Suggested next steps" are excluded
- _is_in_references_section: citations after ## References are handled
- compute_citation_precision: d03/d07 style (Suggested next steps) get excluded
- compute_citation_precision: d04/d05 style (References-only) count as valid
- compute_groundedness: References section sentences are skipped
"""

import pytest

from evaluator import (
    _is_in_references_section,
    _is_in_suggested_next_steps,
    compute_citation_precision,
    compute_groundedness,
)


# ---------------------------------------------------------------------------
# _is_in_suggested_next_steps
# ---------------------------------------------------------------------------


def test_is_in_suggested_next_steps_true():
    """Citation on a line starting with 'Suggested next steps:' returns True."""
    answer = "No sufficient evidence found in the corpus.\n\nSuggested next steps: [arxiv_2501_08774, arxiv_2501_08774_chunk_0001], [arxiv_2502_08108, arxiv_2502_08108_chunk_0006]"
    # First citation: [arxiv_2501_08774, arxiv_2501_08774_chunk_0001]
    cite_start = answer.find("[arxiv_2501_08774")
    cite_end = answer.find("chunk_0001]") + len("chunk_0001]")
    assert _is_in_suggested_next_steps(answer, cite_start, cite_end) is True


def test_is_in_suggested_next_steps_false():
    """Citation in body text returns False."""
    answer = "AI tools improve productivity [arxiv_2409_18048, arxiv_2409_18048_chunk_0003] according to the corpus."
    cite_start = answer.find("[arxiv_2409_18048")
    cite_end = answer.find("chunk_0003]") + len("chunk_0003]")
    assert _is_in_suggested_next_steps(answer, cite_start, cite_end) is False


def test_is_in_suggested_next_steps_case_insensitive():
    """'Suggested next steps' is case-insensitive."""
    answer = "SUGGESTED NEXT STEPS: [arxiv_2501_08774, arxiv_2501_08774_chunk_0001]"
    cite_start = answer.find("[arxiv_2501_08774")
    cite_end = answer.find("chunk_0001]") + len("chunk_0001]")
    assert _is_in_suggested_next_steps(answer, cite_start, cite_end) is True


# ---------------------------------------------------------------------------
# _is_in_references_section
# ---------------------------------------------------------------------------


def test_is_in_references_section_true():
    """Citation after ## References returns True."""
    answer = "The key phases are X, Y, Z.\n\n## References\n- [arxiv_2106_09323, arxiv_2106_09323_chunk_0026]"
    cite_start = answer.find("[arxiv_2106_09323")
    assert _is_in_references_section(answer, cite_start) is True


def test_is_in_references_section_false():
    """Citation before ## References returns False."""
    answer = "The key phases are [arxiv_2106_09323, arxiv_2106_09323_chunk_0026] X, Y, Z.\n\n## References"
    cite_start = answer.find("[arxiv_2106_09323")
    assert _is_in_references_section(answer, cite_start) is False


def test_is_in_references_section_reference_singular():
    """## Reference (no s) also matches."""
    answer = "Text.\n\n## Reference\n- [ijnrd_2024, ijnrd_2024_chunk_0044]"
    cite_start = answer.find("[ijnrd_2024")
    assert _is_in_references_section(answer, cite_start) is True


# ---------------------------------------------------------------------------
# compute_citation_precision — Suggested next steps excluded
# ---------------------------------------------------------------------------


def test_citation_precision_excludes_suggested_next_steps():
    """d03/d07 style: citations in Suggested next steps are excluded; precision is 1.0 when no other citations."""
    answer = """No sufficient evidence found in the corpus.

Suggested next steps: [arxiv_2501_08774, arxiv_2501_08774_chunk_0001], [arxiv_2502_08108, arxiv_2502_08108_chunk_0006]"""
    chunk_lookup = {
        "arxiv_2501_08774_chunk_0001": "Some text.",
        "arxiv_2502_08108_chunk_0006": "Other text.",
    }
    result = compute_citation_precision(answer, chunk_lookup)
    # All citations are in Suggested next steps → excluded → total=0, precision=1.0
    assert result["total_citations"] == 0
    assert result["precision"] == 1.0
    assert result["valid_citations"] == 0


def test_citation_precision_references_only_count_valid():
    """d04/d05 style: citations in ## References with valid chunk_id count as valid."""
    answer = """The key phases include requirement analysis, design, implementation.

## References
- [arxiv_2106_09323, arxiv_2106_09323_chunk_0026]
- [arxiv_2106_09323, arxiv_2106_09323_chunk_0055]"""
    chunk_lookup = {
        "arxiv_2106_09323_chunk_0026": "The waterfall model has phases: requirements, design, implementation.",
        "arxiv_2106_09323_chunk_0055": "Software lifecycle phases.",
    }
    result = compute_citation_precision(answer, chunk_lookup)
    # Both citations are in References section → count as valid (chunk exists)
    assert result["total_citations"] == 2
    assert result["valid_citations"] == 2
    assert result["precision"] == 1.0


# ---------------------------------------------------------------------------
# compute_groundedness — References section skipped
# ---------------------------------------------------------------------------


def test_groundedness_skips_references_section():
    """References block and citation lines are not counted as ungrounded."""
    answer = """The key phases are requirement analysis, design, implementation.

## References
- [arxiv_2106_09323, arxiv_2106_09323_chunk_0026]
- [arxiv_2106_09323, arxiv_2106_09323_chunk_0055]
- International Journal of Novel Research and Development.
- https://www.ijnrd.org/"""
    retrieved = [
        {"text": "The waterfall model has phases: requirements, design, implementation."},
    ]
    result = compute_groundedness(answer, retrieved)
    # "The key phases are..." should be grounded; References lines should be skipped
    assert result["score"] >= 0.5
    # References lines should not appear in ungrounded
    ungrounded = " ".join(result.get("ungrounded", []))
    assert "## References" not in ungrounded or result["grounded_sentences"] > 0


def test_groundedness_skips_no_sufficient_evidence():
    """No sufficient evidence sentence is still treated as abstention."""
    answer = "No sufficient evidence found in the corpus.\n\nSuggested next steps: query1, query2"
    retrieved = [{"text": "Unrelated text."}]
    result = compute_groundedness(answer, retrieved)
    assert result["score"] >= 0.5
    assert result["grounded_sentences"] >= 1
