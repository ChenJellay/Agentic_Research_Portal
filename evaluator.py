"""
Evaluation framework for the Phase 2 RAG pipeline.

Metrics
-------
1. **Groundedness / Faithfulness**: checks whether claims in the model's
   answer are supported by the retrieved context chunks (heuristic overlap).
2. **Citation Precision**: fraction of inline citations ``[source_id, chunk_id]``
   whose chunk text actually contains supporting evidence for the surrounding
   sentence in the answer.

The evaluator can also generate a Markdown report with aggregate scores,
per-query breakdowns, and representative failure cases.
"""

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import get_path_config, get_rag_config
from logger_config import setup_logger
from rag_prompts import extract_citations

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _sentence_split(text: str) -> List[str]:
    """Split text into sentences (simple heuristic)."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _overlap_score(sentence: str, chunk_text: str) -> float:
    """
    Jaccard word overlap between a sentence and a chunk.
    Higher means the sentence is better supported by the chunk.
    """
    s_words = set(sentence.lower().split())
    c_words = set(chunk_text.lower().split())
    if not s_words or not c_words:
        return 0.0
    return len(s_words & c_words) / len(s_words | c_words)


# Pattern to match "No sufficient evidence" abstention
_NO_EVIDENCE_RE = re.compile(
    r"\bno\s+sufficient\s+evidence\s+found\s+in\s+the\s+corpus\b",
    re.IGNORECASE
)


def compute_groundedness(
    answer: str,
    retrieved_chunks: List[Dict[str, Any]],
    threshold: float = 0.12,
) -> Dict[str, Any]:
    """
    Heuristic groundedness: for each sentence in the answer, check whether
    at least one retrieved chunk has sufficient word overlap.

    The "No sufficient evidence found in the corpus" sentence is treated
    as a non-claim (not penalized as ungrounded) when the model abstains.

    Returns:
        {
            "score": float,           # fraction of grounded sentences
            "total_sentences": int,
            "grounded_sentences": int,
            "ungrounded": [str, ...]  # sentences with no supporting chunk
        }
    """
    sentences = _sentence_split(answer)
    if not sentences:
        return {"score": 1.0, "total_sentences": 0, "grounded_sentences": 0, "ungrounded": []}

    chunk_texts = [c.get("text", "") for c in retrieved_chunks]
    grounded = 0
    ungrounded: List[str] = []

    for sent in sentences:
        # Skip very short sentences (headers, citations-only lines)
        if len(sent.split()) < 4:
            grounded += 1
            continue
        # Treat "No sufficient evidence" as correct abstention — not ungrounded
        if _NO_EVIDENCE_RE.search(sent):
            grounded += 1
            continue
        best_overlap = max((_overlap_score(sent, ct) for ct in chunk_texts), default=0.0)
        if best_overlap >= threshold:
            grounded += 1
        else:
            ungrounded.append(sent)

    score = grounded / len(sentences) if sentences else 1.0
    return {
        "score": round(score, 4),
        "total_sentences": len(sentences),
        "grounded_sentences": grounded,
        "ungrounded": ungrounded[:5],  # cap for readability
    }


_CITATION_PATTERN = re.compile(r"\[([^,\]]+),\s*([^\]]+)\]")


def _find_enclosing_sentence(answer: str, cite_start: int, cite_end: int) -> str:
    """
    Find the sentence that contains the citation at [cite_start, cite_end].
    Uses character boundaries: sentence = text from last .!? before cite to next .!? after cite.
    """
    # Start of sentence: after last sentence-ending before cite
    before = answer[:cite_start]
    last_end = max(
        before.rfind("."),
        before.rfind("!"),
        before.rfind("?"),
    )
    sent_start = (last_end + 1) if last_end >= 0 else 0

    # End of sentence: include up to and past the next .!? after cite
    after = answer[cite_end:]
    for i, c in enumerate(after):
        if c in ".!?":
            sent_end = cite_end + i + 1
            break
    else:
        sent_end = len(answer)

    return answer[sent_start:sent_end].strip()


def compute_citation_precision(
    answer: str,
    chunk_lookup: Dict[str, str],
    threshold: float = 0.08,
) -> Dict[str, Any]:
    """
    For each inline citation ``[source_id, chunk_id]`` in the answer,
    check whether the cited chunk text has word-overlap with the
    surrounding sentence.

    Enclosing sentence: the sentence that contains the citation (by character
    position), so citations at end of sentence are correctly attributed.

    Args:
        chunk_lookup: ``{chunk_id: text}`` mapping.

    Returns:
        {
            "precision": float,
            "total_citations": int,
            "valid_citations": int,
            "invalid_citations": [{"chunk_id": ..., "sentence": ...}, ...]
        }
    """
    valid = 0
    invalid_list: List[Dict[str, Any]] = []
    total = 0

    for m in _CITATION_PATTERN.finditer(answer):
        total += 1
        source_id = m.group(1).strip()
        cid = m.group(2).strip()
        cite_start, cite_end = m.span()

        chunk_text = chunk_lookup.get(cid, "")
        if not chunk_text:
            invalid_list.append({"chunk_id": cid, "reason": "chunk_id not found in index"})
            continue

        enclosing = _find_enclosing_sentence(answer, cite_start, cite_end)
        if not enclosing:
            enclosing = answer

        overlap = _overlap_score(enclosing, chunk_text)
        if overlap >= threshold:
            valid += 1
        else:
            invalid_list.append({
                "chunk_id": cid,
                "reason": f"low overlap ({overlap:.3f}) with enclosing sentence",
            })

    precision = valid / total if total else 1.0
    return {
        "precision": round(precision, 4),
        "total_citations": total,
        "valid_citations": valid,
        "invalid_citations": invalid_list[:5],
    }


# ---------------------------------------------------------------------------
# Confidence / evidence-strength score
# ---------------------------------------------------------------------------

def _compute_confidence(
    groundedness: Dict[str, Any],
    citation_prec: Dict[str, Any],
    answer: str,
) -> Tuple[float, str]:
    """
    Compute per-response confidence score (0–1) and tier.

    Uses composite: 0.6 * groundedness + 0.4 * citation_precision.
    Special case: if answer is only "No sufficient evidence" (+ refs), score 0.5 (abstained).
    """
    # Abstention: model correctly said no evidence
    if "no sufficient evidence" in answer.lower() and groundedness.get("total_sentences", 0) <= 2:
        return (0.5, "abstained")

    g = groundedness.get("score", 0.0)
    c = citation_prec.get("precision", 0.0)
    confidence = round(0.6 * g + 0.4 * c, 4)

    if g >= 0.6 and c >= 0.6:
        tier = "high"
    elif g >= 0.4 or c >= 0.4:
        tier = "medium"
    else:
        tier = "low"
    return (confidence, tier)


# ---------------------------------------------------------------------------
# Single-query evaluation
# ---------------------------------------------------------------------------

def evaluate_single(
    query_record: Dict[str, Any],
    answer: str,
    retrieved_chunks: List[Dict[str, Any]],
    chunk_lookup: Dict[str, str],
) -> Dict[str, Any]:
    """
    Evaluate a single query-answer pair.

    Returns a dict with groundedness, citation precision, and confidence results.
    """
    groundedness = compute_groundedness(answer, retrieved_chunks)
    citation_prec = compute_citation_precision(answer, chunk_lookup)
    confidence, confidence_tier = _compute_confidence(groundedness, citation_prec, answer)

    return {
        "query_id": query_record.get("id", ""),
        "query_type": query_record.get("type", ""),
        "query": query_record.get("query", ""),
        "groundedness": groundedness,
        "citation_precision": citation_prec,
        "confidence": confidence,
        "confidence_tier": confidence_tier,
        "answer_length": len(answer),
    }


# ---------------------------------------------------------------------------
# Batch evaluation (runs queries through the RAG pipeline)
# ---------------------------------------------------------------------------

def run_evaluation(
    run_query_fn,  # callable(query: str) -> Tuple[str, List[Dict]]
    queries_path: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    chunk_lookup: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Run the full evaluation suite.

    Args:
        run_query_fn: a callable ``(query) -> (answer, retrieved_chunks)``
            that executes the RAG pipeline for a single query.
        queries_path: path to eval/queries.json.
        results_dir: directory to save results.
        chunk_lookup: ``{chunk_id: text}`` for citation checking.

    Returns:
        Aggregate evaluation summary.
    """
    paths = get_path_config()
    queries_path = queries_path or paths.eval_queries_path
    results_dir = results_dir or paths.eval_results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load queries
    with open(queries_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    queries = data.get("queries", [])
    logger.info(f"Loaded {len(queries)} evaluation queries from {queries_path}")

    # If no chunk_lookup provided, try to build one from index metadata
    if chunk_lookup is None:
        chunk_lookup = _build_chunk_lookup()

    per_query_results: List[Dict[str, Any]] = []
    failure_cases: List[Dict[str, Any]] = []

    for q in queries:
        qtext = q.get("query", "")
        qid = q.get("id", "unknown")
        logger.info(f"Evaluating [{qid}]: {qtext[:60]}…")

        if not qtext.strip():
            # Edge case: empty query
            per_query_results.append({
                "query_id": qid,
                "query_type": q.get("type", ""),
                "query": qtext,
                "skipped": True,
                "reason": "empty query",
            })
            continue

        try:
            answer, retrieved_chunks = run_query_fn(qtext)
            result = evaluate_single(q, answer, retrieved_chunks, chunk_lookup)
            per_query_results.append(result)

            # Track failure cases
            if result["groundedness"]["score"] < 0.5 or result["citation_precision"]["precision"] < 0.5:
                failure_cases.append({
                    "query_id": qid,
                    "query": qtext,
                    "groundedness_score": result["groundedness"]["score"],
                    "citation_precision": result["citation_precision"]["precision"],
                    "ungrounded_samples": result["groundedness"].get("ungrounded", [])[:2],
                    "answer_snippet": answer[:300],
                })
        except Exception as e:
            logger.error(f"Error evaluating [{qid}]: {e}")
            per_query_results.append({
                "query_id": qid,
                "query_type": q.get("type", ""),
                "query": qtext,
                "error": str(e),
            })
            failure_cases.append({
                "query_id": qid,
                "query": qtext,
                "error": str(e),
            })

    # Aggregate
    scored = [r for r in per_query_results if "groundedness" in r]
    avg_ground = (
        sum(r["groundedness"]["score"] for r in scored) / len(scored)
        if scored else 0.0
    )
    avg_cite = (
        sum(r["citation_precision"]["precision"] for r in scored) / len(scored)
        if scored else 0.0
    )
    avg_confidence = (
        sum(r["confidence"] for r in scored) / len(scored)
        if scored else 0.0
    )

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_queries": len(queries),
        "evaluated": len(scored),
        "skipped": len(queries) - len(scored),
        "avg_groundedness": round(avg_ground, 4),
        "avg_citation_precision": round(avg_cite, 4),
        "avg_confidence": round(avg_confidence, 4),
        "failure_cases_count": len(failure_cases),
        "failure_cases": failure_cases[:5],
    }

    # Save detailed results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = results_dir / f"eval_{ts}.json"
    with open(detail_path, "w", encoding="utf-8") as fh:
        json.dump({
            "summary": summary,
            "per_query": per_query_results,
        }, fh, indent=2, ensure_ascii=False)
    logger.info(f"Evaluation results saved → {detail_path}")

    # Generate markdown report
    report_path = results_dir / f"eval_report_{ts}.md"
    _write_report(report_path, summary, per_query_results, failure_cases)
    logger.info(f"Evaluation report saved → {report_path}")

    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_chunk_lookup() -> Dict[str, str]:
    """Build {chunk_id: text} from the index metadata file."""
    paths = get_path_config()
    meta_path = paths.chunk_metadata_path
    if not meta_path.exists():
        logger.warning("Chunk metadata not found; citation precision may be inaccurate.")
        return {}
    with open(meta_path, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    return {m["chunk_id"]: m.get("text", "") for m in metadata}


def _write_report(
    path: Path,
    summary: Dict[str, Any],
    per_query: List[Dict[str, Any]],
    failure_cases: List[Dict[str, Any]],
) -> None:
    """Write a human-readable Markdown evaluation report."""
    lines = [
        "# RAG Evaluation Report",
        "",
        f"**Date**: {summary['timestamp']}",
        "",
        "## Aggregate Metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Queries | {summary['total_queries']} |",
        f"| Evaluated | {summary['evaluated']} |",
        f"| Skipped | {summary['skipped']} |",
        f"| Avg Groundedness | {summary['avg_groundedness']:.4f} |",
        f"| Avg Citation Precision | {summary['avg_citation_precision']:.4f} |",
        f"| Avg Confidence | {summary.get('avg_confidence', 0):.4f} |",
        f"| Failure Cases | {summary['failure_cases_count']} |",
        "",
        "## Per-Query Results",
        "",
    ]

    # Group by type
    for qtype in ["direct", "synthesis", "edge_case", "stress_test"]:
        typed = [r for r in per_query if r.get("query_type") == qtype]
        if not typed:
            continue
        lines.append(f"### {qtype.replace('_', ' ').title()} Queries")
        lines.append("")
        lines.append("| ID | Groundedness | Citation Prec. | Confidence | Answer Len |")
        lines.append("|----|-------------|----------------|------------|-----------|")
        for r in typed:
            if "groundedness" in r:
                g = r["groundedness"]["score"]
                c = r["citation_precision"]["precision"]
                conf = r.get("confidence", 0)
                tier = r.get("confidence_tier", "")
                a = r.get("answer_length", 0)
                lines.append(f"| {r['query_id']} | {g:.3f} | {c:.3f} | {conf:.3f} ({tier}) | {a} |")
            elif r.get("skipped"):
                lines.append(f"| {r['query_id']} | SKIPPED | — | — | — |")
            else:
                lines.append(f"| {r['query_id']} | ERROR | — | — | — |")
        lines.append("")

    # Failure cases
    lines.append("## Representative Failure Cases")
    lines.append("")
    if not failure_cases:
        lines.append("No failure cases detected.")
    else:
        for i, fc in enumerate(failure_cases[:5], 1):
            lines.append(f"### Failure {i}: {fc.get('query_id', 'N/A')}")
            lines.append("")
            lines.append(f"**Query**: {fc.get('query', 'N/A')}")
            lines.append("")
            if "error" in fc:
                lines.append(f"**Error**: {fc['error']}")
            else:
                lines.append(f"**Groundedness**: {fc.get('groundedness_score', 'N/A')}")
                lines.append(f"**Citation Precision**: {fc.get('citation_precision', 'N/A')}")
                ungrounded = fc.get("ungrounded_samples", [])
                if ungrounded:
                    lines.append("")
                    lines.append("**Ungrounded sentences**:")
                    for u in ungrounded:
                        lines.append(f"> {u}")
                snippet = fc.get("answer_snippet", "")
                if snippet:
                    lines.append("")
                    lines.append(f"**Answer snippet**: {snippet}…")
            lines.append("")

    # Interpretation
    lines.extend([
        "## Interpretation",
        "",
        "- **Groundedness** measures what fraction of answer sentences are supported by at least one retrieved chunk (word-overlap heuristic, threshold=0.12). The \"No sufficient evidence\" sentence is treated as correct abstention.",
        "- **Citation Precision** measures what fraction of inline `[source_id, chunk_id]` citations point to chunks that actually support the enclosing sentence (threshold=0.08).",
        "- **Confidence / evidence-strength** is a composite: 0.6×groundedness + 0.4×citation_precision. Tier: high (both ≥0.6), medium (either ≥0.4), low (otherwise). Answers that abstain with \"No sufficient evidence\" receive 0.5 (correctly abstained).",
        "- Failure cases highlight queries where either groundedness or citation precision fell below 0.5.",
        "",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")
