"""
Artifact generator — Phase 3.

Produces research artifacts from threads:
- Evidence table: Claim | Evidence snippet | Citation | Confidence | Notes
- Annotated bibliography: 8–12 sources with (claim, method, limitations, why it matters)
- Synthesis memo: 800–1200 words with inline citations and reference list
"""

import json
import re
from typing import Any, Dict, List

from config import get_model_config, get_rag_config
from logger_config import setup_logger

logger = setup_logger(__name__)

_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        from mlx_agent import MLXAgent
        _agent = MLXAgent(get_model_config())
        _agent.initialize_model()
    return _agent


# ---------------------------------------------------------------------------
# Evidence Table
# ---------------------------------------------------------------------------

EVIDENCE_TABLE_PROMPT = """\
Given the following research question, answer, and retrieved chunks, extract each factual claim \
from the answer and map it to supporting evidence.

QUESTION: {query}

ANSWER:
{answer}

RETRIEVED CHUNKS:
{chunks_text}

Produce a JSON object with key "rows" containing a list of objects, each with:
- claim: the factual claim from the answer
- evidence_snippet: a complete quote from a chunk that supports it (include full sentences, do not truncate)
- source_id: source_id of the chunk
- chunk_id: chunk_id of the chunk
- confidence: 0.0 to 1.0 (how well the evidence supports the claim)
- notes: optional brief note

Output ONLY valid JSON, no other text.
"""


def generate_evidence_table(thread: Dict[str, Any]) -> Dict[str, Any]:
    """Generate evidence table from thread."""
    query = thread.get("query", "")
    answer = thread.get("answer", "")
    chunks = thread.get("retrieved_chunks", [])

    chunks_text = "\n\n".join(
        f"[{c.get('chunk_id')}] {c.get('text', '')[:1200]}"
        for c in chunks[:10]
    )

    prompt = EVIDENCE_TABLE_PROMPT.format(
        query=query,
        answer=answer,
        chunks_text=chunks_text,
    )

    agent = _get_agent()
    rag_cfg = get_rag_config()
    raw = agent.generate_response(
        prompt,
        max_tokens=rag_cfg.rag_max_tokens,
        temperature=0.2,
    )

    rows = _parse_json_list(raw, "rows")
    if not rows:
        rows = [{"claim": "No claims extracted", "evidence_snippet": "-", "source_id": "-", "chunk_id": "-", "confidence": 0, "notes": "LLM parsing failed"}]

    markdown = _evidence_table_to_markdown(rows)
    return {"rows": rows, "markdown": markdown}


def _evidence_table_to_markdown(rows: List[Dict]) -> str:
    lines = ["| Claim | Evidence snippet | Citation | Confidence | Notes |", "|-------|------------------|----------|------------|-------|"]
    for r in rows:
        cite = f"{r.get('source_id', '')}, {r.get('chunk_id', '')}"
        # Preserve full evidence (up to 600 chars) to avoid cutting off sentences
        snippet = r.get("evidence_snippet", "")
        if len(snippet) > 600:
            snippet = snippet[:597] + "..."
        lines.append(f"| {r.get('claim', '')} | {snippet} | {cite} | {r.get('confidence', 0)} | {r.get('notes', '')} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Annotated Bibliography
# ---------------------------------------------------------------------------

ANNOTATED_BIB_PROMPT = """\
For each of the following sources (from retrieved chunks), produce a brief annotated entry.

SOURCES (with metadata):
{source_list}

CHUNK EXCERPTS:
{chunks_text}

Produce a JSON object with key "entries" containing a list of objects, one per source, each with:
- source_id: the source identifier
- title: from metadata
- claim: main claim or thesis of this source (1-2 sentences)
- method: research method or approach used
- limitations: key limitations mentioned or implied
- why_it_matters: relevance to the research question

Output ONLY valid JSON, no other text.
"""


def generate_annotated_bib(thread: Dict[str, Any]) -> Dict[str, Any]:
    """Generate annotated bibliography from thread."""
    chunks = thread.get("retrieved_chunks", [])
    source_meta = thread.get("source_metadata", {})

    source_ids = list(dict.fromkeys(c.get("source_id") for c in chunks if c.get("source_id")))
    source_list = []
    for sid in source_ids[:12]:
        m = source_meta.get(sid, {})
        source_list.append(f"- {sid}: {m.get('title', '')} ({m.get('authors', [])}, {m.get('year', '')})")

    chunks_text = "\n\n".join(
        f"[{c.get('chunk_id')}] {c.get('text', '')[:800]}"
        for c in chunks[:20]
    )

    prompt = ANNOTATED_BIB_PROMPT.format(
        source_list="\n".join(source_list),
        chunks_text=chunks_text,
    )

    agent = _get_agent()
    rag_cfg = get_rag_config()
    raw = agent.generate_response(
        prompt,
        max_tokens=rag_cfg.rag_max_tokens,
        temperature=0.2,
    )

    entries = _parse_json_list(raw, "entries")
    if not entries:
        entries = [{"source_id": s, "title": source_meta.get(s, {}).get("title", ""), "claim": "-", "method": "-", "limitations": "-", "why_it_matters": "-"} for s in source_ids[:8]]

    markdown = _annotated_bib_to_markdown(entries)
    return {"entries": entries, "markdown": markdown}


def _annotated_bib_to_markdown(entries: List[Dict]) -> str:
    lines = []
    for e in entries:
        lines.append(f"### {e.get('source_id', '')}: {e.get('title', '')}")
        lines.append(f"- **Claim**: {e.get('claim', '')}")
        lines.append(f"- **Method**: {e.get('method', '')}")
        lines.append(f"- **Limitations**: {e.get('limitations', '')}")
        lines.append(f"- **Why it matters**: {e.get('why_it_matters', '')}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Synthesis Memo
# ---------------------------------------------------------------------------

SYNTHESIS_MEMO_PROMPT = """\
Expand the following research answer into a formal synthesis memo of 800–1200 words.

QUESTION: {query}

ORIGINAL ANSWER:
{answer}

RETRIEVED CHUNKS (for reference):
{chunks_text}

SOURCE METADATA:
{source_meta}

Requirements:
1. Write in formal academic style.
2. Use inline citations as [source_id, chunk_id] for every factual claim.
3. End with a "## References" section listing each cited source with full metadata.
4. Length: 800–1200 words.
5. Do NOT invent citations; use only chunk_ids from the chunks above.
"""


def generate_synthesis_memo(thread: Dict[str, Any]) -> Dict[str, Any]:
    """Generate synthesis memo from thread."""
    query = thread.get("query", "")
    answer = thread.get("answer", "")
    chunks = thread.get("retrieved_chunks", [])
    source_meta = thread.get("source_metadata", {})

    chunks_text = "\n\n".join(
        f"[Chunk] {c.get('chunk_id')} (source: {c.get('source_id')})\n{c.get('text', '')[:600]}"
        for c in chunks[:10]
    )

    meta_lines = [f"- {sid}: {m.get('title', '')} ({m.get('authors', [])}, {m.get('year', '')}). {m.get('venue', '')}. {m.get('link', '')}" for sid, m in source_meta.items()]
    source_meta_text = "\n".join(meta_lines) if meta_lines else "(none)"

    prompt = SYNTHESIS_MEMO_PROMPT.format(
        query=query,
        answer=answer,
        chunks_text=chunks_text,
        source_meta=source_meta_text,
    )

    agent = _get_agent()
    rag_cfg = get_rag_config()
    raw = agent.generate_response(
        prompt,
        max_tokens=2500,
        temperature=0.3,
    )

    return {"content": raw, "markdown": raw}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json_list(raw: str, key: str) -> List[Dict]:
    """Extract a list from JSON output, handling common LLM artifacts."""
    # Try to find JSON block
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            data = json.loads(m.group(0))
            return data.get(key, [])
        except json.JSONDecodeError:
            pass
    return []
