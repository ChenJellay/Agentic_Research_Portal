"""
Section-aware text chunker for RAG corpus.

Strategy
--------
1. Process each section from the structured PDF extraction independently.
2. If a section fits within ``chunk_size`` tokens, keep it as a single chunk.
3. If it exceeds ``chunk_size``, split at sentence boundaries with ``overlap``
   tokens of overlap between consecutive chunks.
4. Every chunk carries rich metadata for traceability.

Token counting uses ``tiktoken`` (cl100k_base) for accurate estimates that
align with transformer model tokenizers.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import tiktoken

from config import get_rag_config
from logger_config import setup_logger

logger = setup_logger(__name__)

# Sentence boundary regex — handles ". ", "! ", "? " and similar.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

# Shared tokenizer (lazy-loaded)
_enc: Optional[tiktoken.Encoding] = None


def _get_encoder() -> tiktoken.Encoding:
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding("cl100k_base")
    return _enc


def _count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


def count_tokens(text: str) -> int:
    """
    Count tokens in text (cl100k_base). Use for RAG context budgeting and prompt construction.
    Shared with chunker so token counts are consistent across ingest and query time.
    """
    return _count_tokens(text)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Return the prefix of text that fits within max_tokens (trimmed from end)."""
    if max_tokens <= 0:
        return ""
    enc = _get_encoder()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])


def _make_chunk_id(source_id: str, idx: int) -> str:
    """Deterministic chunk ID: source_id + sequential index."""
    return f"{source_id}_chunk_{idx:04d}"


# ---------------------------------------------------------------------------
# Core chunking
# ---------------------------------------------------------------------------

def chunk_sections(
    source_id: str,
    sections: List[Dict[str, Any]],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Chunk a list of sections (from ``extract_sections_from_pdf``) into
    retrieval-sized pieces.

    Args:
        source_id: The manifest source_id for provenance.
        sections: List of ``{"section", "text", "page_start", "page_end"}``.
        chunk_size: Max tokens per chunk (default from RAGConfig).
        chunk_overlap: Overlap tokens between consecutive chunks.

    Returns:
        List of chunk dicts::

            {
                "chunk_id": "arxiv_2409_18048_chunk_0001",
                "source_id": "arxiv_2409_18048",
                "section": "Introduction",
                "page_start": 1,
                "page_end": 2,
                "char_start": 0,
                "char_end": 1234,
                "text": "...",
                "token_count": 487
            }
    """
    cfg = get_rag_config()
    chunk_size = chunk_size or cfg.chunk_size
    chunk_overlap = chunk_overlap or cfg.chunk_overlap

    chunks: List[Dict[str, Any]] = []
    global_idx = 0

    for sec in sections:
        sec_text = sec["text"].strip()
        if not sec_text:
            continue

        sec_tokens = _count_tokens(sec_text)

        if sec_tokens <= chunk_size:
            # Whole section fits in one chunk
            chunks.append({
                "chunk_id": _make_chunk_id(source_id, global_idx),
                "source_id": source_id,
                "section": sec["section"],
                "page_start": sec["page_start"],
                "page_end": sec["page_end"],
                "char_start": 0,
                "char_end": len(sec_text),
                "text": sec_text,
                "token_count": sec_tokens,
            })
            global_idx += 1
        else:
            # Split section at sentence boundaries with overlap
            sub_chunks = _split_text_into_chunks(sec_text, chunk_size, chunk_overlap)
            char_cursor = 0
            for sub in sub_chunks:
                # Locate sub-text position in original section
                pos = sec_text.find(sub[:60], char_cursor)
                if pos == -1:
                    pos = char_cursor
                chunks.append({
                    "chunk_id": _make_chunk_id(source_id, global_idx),
                    "source_id": source_id,
                    "section": sec["section"],
                    "page_start": sec["page_start"],
                    "page_end": sec["page_end"],
                    "char_start": pos,
                    "char_end": pos + len(sub),
                    "text": sub,
                    "token_count": _count_tokens(sub),
                })
                global_idx += 1
                char_cursor = pos + len(sub) - (chunk_overlap * 4)  # rough char estimate

    logger.info(
        f"Chunked source {source_id}: {len(sections)} section(s) → {len(chunks)} chunk(s)"
    )
    return chunks


def _split_text_into_chunks(
    text: str, chunk_size: int, overlap: int
) -> List[str]:
    """
    Split text at sentence boundaries respecting token limits.
    """
    sentences = _SENTENCE_SPLIT_RE.split(text)
    if not sentences:
        return [text]

    chunks: List[str] = []
    current_sents: List[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = _count_tokens(sent)

        if current_tokens + sent_tokens <= chunk_size:
            current_sents.append(sent)
            current_tokens += sent_tokens
        else:
            if current_sents:
                chunks.append(" ".join(current_sents))

            # Overlap: carry some trailing sentences forward
            overlap_sents: List[str] = []
            overlap_tokens = 0
            for s in reversed(current_sents):
                s_tok = _count_tokens(s)
                if overlap_tokens + s_tok > overlap:
                    break
                overlap_sents.insert(0, s)
                overlap_tokens += s_tok

            current_sents = overlap_sents + [sent]
            current_tokens = overlap_tokens + sent_tokens

    if current_sents:
        chunks.append(" ".join(current_sents))

    return chunks


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_chunks(chunks: List[Dict[str, Any]], dest_path: Path) -> None:
    """Append chunks to a JSONL file."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "a", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(chunks)} chunk(s) → {dest_path}")


def load_all_chunks(chunks_path: Path) -> List[Dict[str, Any]]:
    """Load all chunks from a JSONL file."""
    if not chunks_path.exists():
        return []
    chunks: List[Dict[str, Any]] = []
    with open(chunks_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks
