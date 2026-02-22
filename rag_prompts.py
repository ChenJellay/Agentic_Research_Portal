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
3. Place each citation immediately after the claim it supports — one sentence, \
one citation when possible. Use the format [source_id, chunk_id] \
(e.g., [arxiv_2409_18048, arxiv_2409_18048_chunk_0003]). Do NOT use generic \
phrases like "according to the corpus" without a specific [source_id, chunk_id].
4. If the context chunks do NOT contain enough evidence to answer the \
question, say: "No sufficient evidence found in the corpus." Then add \
a line: "Suggested next steps: [query1], [query2]" with 1–2 alternative \
search queries the user could try to find relevant evidence.
5. If sources present conflicting evidence, explicitly flag the conflict \
and cite both sides.
6. Do NOT invent or hallucinate citations. Every cited chunk_id must come \
from the context below.
7. End your answer with a "## References" section listing every source you \
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

_CITATION_RE = re.compile(r"\[([^,\]]+),\s*([^\]]+)\]")


def extract_citations(text: str) -> List[Dict[str, str]]:
    """
    Extract inline citations of the form ``[source_id, chunk_id]``.

    Returns a list of ``{"source_id": ..., "chunk_id": ...}`` dicts.
    """
    citations: List[Dict[str, str]] = []
    for match in _CITATION_RE.finditer(text):
        citations.append({
            "source_id": match.group(1).strip(),
            "chunk_id": match.group(2).strip(),
        })
    return citations


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
