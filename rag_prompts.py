"""
RAG prompt templates — Phase 2.

Enhancement #2: Structured Citations.

Provides prompt construction that instructs the model to:
  - Answer using ONLY the provided context chunks.
  - Cite sources inline as ``[source_id, chunk_id]``.
  - End with a "References" section listing full source metadata.
  - Explicitly state "No evidence found in the corpus" when retrieval
    is insufficient.
  - Flag conflicting evidence across sources.
"""

from typing import Any, Dict, List, Optional

from token_utils import count_tokens

PROMPT_TEMPLATE_VERSION = "v1"

# ---------------------------------------------------------------------------
# System prompt (prepended to every RAG query)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a research assistant that answers questions based ONLY on the \
provided context chunks from a curated corpus of academic papers and \
technical reports on AI-augmented software development.

RULES — follow these strictly:
1. Use ONLY information present in the context chunks below to answer.
2. Make only claims that are directly supported by a specific chunk. If you \
cannot support a claim with a specific chunk, do not make that claim.
3. **Citation format**: Use exactly [source_id, chunk_id] with a comma and space \
between them. Examples: [arxiv_2409_18048, arxiv_2409_18048_chunk_0003], \
[ijnrd_2024, ijnrd_2024_chunk_0043]. Do NOT use variants like [source_id=X, \
chunk_id=Y] or (source_id, chunk_id) or bare chunk_ids without source_id.
4. Place each citation immediately after the claim it supports — one sentence, \
one citation when possible. Do NOT use generic phrases like "according to the \
corpus" without a specific [source_id, chunk_id].
5. If the context chunks do NOT contain enough evidence to answer the \
question, say: "No sufficient evidence found in the corpus." Then add \
a line: "Suggested next steps: [query1], [query2]" with 1–2 alternative \
search queries the user could try to find relevant evidence.
6. If sources present conflicting evidence, explicitly flag the conflict \
and cite both sides.
7. Do NOT invent or hallucinate citations. Every cited chunk_id must come \
from the context below.
8. End your answer with a "## References" section listing every source you \
cited, using the metadata provided."""


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text from the end to fit within max_tokens."""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


def build_rag_prompt(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    source_metadata: Dict[str, Dict[str, Any]],
    context_token_budget: Optional[int] = None,
) -> str:
    """
    Construct the full prompt for the generation model.

    Args:
        query: The user's question.
        retrieved_chunks: List of chunk dicts from the retriever, each with
            at least ``chunk_id``, ``source_id``, ``section``, ``text``.
        source_metadata: ``{source_id: manifest_entry}`` for sources that
            appear in retrieved chunks.
        context_token_budget: If provided, trim chunks so total context
            (chunks + metadata + question) fits within this token budget.

    Returns:
        A single string ready to be passed to the model.
    """
    # Build metadata block first (needed for token count)
    meta_lines: List[str] = []
    for sid, meta in source_metadata.items():
        title = meta.get("title", "")
        authors = ", ".join(meta.get("authors", []))
        year = meta.get("year", "")
        venue = meta.get("venue", "")
        link = meta.get("link", "")
        meta_lines.append(
            f"- **{sid}**: {title} ({authors}, {year}). {venue}. {link}"
        )
    meta_block = "\n".join(meta_lines) if meta_lines else "(no metadata available)"

    # Compute non-chunk token usage
    header_block = f"""{SYSTEM_PROMPT}

=== CONTEXT CHUNKS ===

"""
    footer_block = f"""

=== SOURCE METADATA ===

{meta_block}

=== QUESTION ===

{query}

=== ANSWER ===
"""
    fixed_tokens = count_tokens(header_block) + count_tokens(footer_block)
    chunk_budget = (context_token_budget - fixed_tokens) if context_token_budget else None

    # Build context section (with optional truncation)
    context_parts: List[str] = []
    used_tokens = 0
    for i, chunk in enumerate(retrieved_chunks, 1):
        cid = chunk.get("chunk_id", f"chunk_{i}")
        sid = chunk.get("source_id", "unknown")
        section = chunk.get("section", "")
        text = chunk.get("text", "")
        header = f"[Chunk {i}]  chunk_id={cid}  source_id={sid}"
        if section:
            header += f"  section=\"{section}\""
        chunk_block = f"{header}\n{text}"
        chunk_tokens = count_tokens(chunk_block)

        if chunk_budget is not None:
            remaining = chunk_budget - used_tokens
            if remaining <= 0:
                break
            if chunk_tokens > remaining:
                # Truncate text to fit; keep header
                text_budget = remaining - count_tokens(header) - 10  # margin
                if text_budget > 0:
                    text = _truncate_text_to_tokens(text, text_budget)
                    chunk_block = f"{header}\n{text}"
                    chunk_tokens = count_tokens(chunk_block)
                else:
                    break
            used_tokens += chunk_tokens

        context_parts.append(chunk_block)

    context_block = "\n\n---\n\n".join(context_parts)

    # Assemble
    prompt = f"""{SYSTEM_PROMPT}

=== CONTEXT CHUNKS ===

{context_block}

=== SOURCE METADATA ===

{meta_block}

=== QUESTION ===

{query}

=== ANSWER ===
"""
    return prompt


# ---------------------------------------------------------------------------
# Citation extraction helpers
# ---------------------------------------------------------------------------

import re

# Standard: [source_id, chunk_id]
_CITATION_RE = re.compile(r"\[([^,\]]+),\s*([^\]]+)\]")

# Variant: [source_id=X, chunk_id=Y] or [source_id = X, chunk_id = Y]
_CITATION_VARIANT_RE = re.compile(
    r"\[?\s*source_id\s*=\s*([^,\]]+)\s*,\s*chunk_id\s*=\s*([^\]]+)\s*\]?",
    re.IGNORECASE
)

# Variant: (source_id, chunk_id) in References
_CITATION_PAREN_RE = re.compile(r"\(\s*([^,\s]+)\s*,\s*([^)\s]+)\s*\)")


def extract_citations(text: str) -> List[Dict[str, str]]:
    """
    Extract inline citations from multiple formats; normalize to standard
    ``[source_id, chunk_id]`` semantics.

    Supported formats:
      - [source_id, chunk_id] (standard)
      - [source_id=X, chunk_id=Y]
      - (source_id, chunk_id) in References or inline

    Returns a list of ``{"source_id": ..., "chunk_id": ...}`` dicts.
    Deduplicates by (source_id, chunk_id).
    """
    seen: set = set()
    citations: List[Dict[str, str]] = []

    def add(sid: str, cid: str) -> None:
        sid, cid = sid.strip(), cid.strip()
        if not sid or not cid:
            return
        if "=" in sid or "=" in cid:
            return  # Variant format parsed by standard regex; skip
        key = (sid, cid)
        if key not in seen:
            seen.add(key)
            citations.append({"source_id": sid, "chunk_id": cid})

    for m in _CITATION_VARIANT_RE.finditer(text):
        add(m.group(1), m.group(2))
    for m in _CITATION_RE.finditer(text):
        add(m.group(1), m.group(2))
    for m in _CITATION_PAREN_RE.finditer(text):
        sid, cid = m.group(1), m.group(2)
        # Only add if it looks like source_id_chunk_XXXX (chunk_id pattern)
        if "_chunk_" in cid or (sid.startswith("arxiv_") or sid.startswith("acm_") or sid.startswith("ijnrd_") or sid.startswith("ijsr_")):
            add(sid, cid)

    return citations


def extract_citations_with_positions(text: str) -> List[Dict[str, Any]]:
    """
    Extract citations with character positions for evaluation.

    Returns list of {"source_id": ..., "chunk_id": ..., "start": int, "end": int}.
    Used by evaluator for enclosing-sentence logic.
    """
    result: List[Dict[str, Any]] = []
    seen: set = set()

    def add(sid: str, cid: str, start: int, end: int) -> None:
        sid, cid = sid.strip(), cid.strip()
        if not sid or not cid:
            return
        key = (sid, cid, start)
        if key not in seen:
            seen.add(key)
            result.append({"source_id": sid, "chunk_id": cid, "start": start, "end": end})

    for m in _CITATION_VARIANT_RE.finditer(text):
        add(m.group(1), m.group(2), m.start(), m.end())
    for m in _CITATION_RE.finditer(text):
        sid, cid = m.group(1).strip(), m.group(2).strip()
        if "=" in sid or "=" in cid:
            continue
        add(sid, cid, m.start(), m.end())
    for m in _CITATION_PAREN_RE.finditer(text):
        sid, cid = m.group(1), m.group(2)
        if "_chunk_" in cid or any(sid.startswith(p) for p in ("arxiv_", "acm_", "ijnrd_", "ijsr_")):
            add(sid, cid, m.start(), m.end())

    return result


def validate_citations(
    citations: List[Dict[str, str]],
    valid_chunk_ids: set,
) -> Dict[str, List[Dict[str, str]]]:
    """
    Split citations into valid and invalid based on known chunk IDs.

    Returns ``{"valid": [...], "invalid": [...]}``.
    """
    valid: List[Dict[str, str]] = []
    invalid: List[Dict[str, str]] = []
    for c in citations:
        if c["chunk_id"] in valid_chunk_ids:
            valid.append(c)
        else:
            invalid.append(c)
    return {"valid": valid, "invalid": invalid}
