"""
End-to-end ingestion pipeline for the RAG corpus.

Orchestrates:  manifest → PDF parse → chunk → embed → build indices.

Can be run standalone::

    python ingest.py

Or from ``rag_pipeline.py ingest``.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from chunker import chunk_sections, load_all_chunks, save_chunks
from config import get_path_config, get_rag_config
from embedder import embed_chunks
from logger_config import setup_logger
from manifest import load_manifest, validate_manifest
from pdf_processor import (
    PDFProcessingError,
    extract_sections_from_pdf,
    save_processed_source,
)
from retriever import BM25Index
from vector_store import VectorStore

logger = setup_logger(__name__)


def run_ingestion(
    manifest_path: Optional[Path] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Execute the full ingestion pipeline.

    Steps:
        1. Load and validate manifest.
        2. Parse each PDF (section-aware).
        3. Chunk all sections.
        4. Embed all chunks.
        5. Build and persist FAISS + BM25 indices.

    Args:
        manifest_path: Override path to manifest.json.
        force: If True, re-process all sources even if processed files exist.

    Returns:
        Summary dict with counts.
    """
    paths = get_path_config()
    paths.ensure_dirs()
    manifest_path = manifest_path or paths.manifest_path

    # ------------------------------------------------------------------
    # 1. Load manifest
    # ------------------------------------------------------------------
    sources = load_manifest(manifest_path)
    if not sources:
        raise RuntimeError(f"Manifest is empty or missing: {manifest_path}")

    problems = validate_manifest(sources)
    if problems:
        for sid, errs in problems.items():
            logger.warning(f"Manifest validation — {sid}: {errs}")

    logger.info(f"Manifest loaded: {len(sources)} source(s)")

    # ------------------------------------------------------------------
    # 2. Parse PDFs → processed JSON
    # ------------------------------------------------------------------
    all_sections: Dict[str, List[Dict[str, Any]]] = {}
    parse_errors: List[str] = []

    for src in sources:
        sid = src["source_id"]
        filename = src["filename"]
        processed_path = paths.processed_dir / f"{sid}.json"

        if processed_path.exists() and not force:
            logger.info(f"Skipping already-processed source: {sid}")
            with open(processed_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            all_sections[sid] = data["sections"]
            continue

        pdf_path = paths.raw_dir / filename
        if not pdf_path.exists():
            logger.error(f"PDF not found for {sid}: {pdf_path}")
            parse_errors.append(sid)
            continue

        try:
            sections = extract_sections_from_pdf(pdf_path)
            save_processed_source(sid, sections, paths.processed_dir)
            all_sections[sid] = sections
        except PDFProcessingError as e:
            logger.error(f"Failed to parse {sid}: {e}")
            parse_errors.append(sid)

    if not all_sections:
        raise RuntimeError("No sources could be parsed; aborting ingestion.")

    logger.info(
        f"Parsed {len(all_sections)} source(s)"
        + (f" ({len(parse_errors)} error(s))" if parse_errors else "")
    )

    # ------------------------------------------------------------------
    # 3. Chunk
    # ------------------------------------------------------------------
    all_chunks: List[Dict[str, Any]] = []
    for sid, sections in all_sections.items():
        chunks = chunk_sections(sid, sections)
        all_chunks.extend(chunks)

    logger.info(f"Total chunks: {len(all_chunks)}")

    # Persist chunks JSONL (overwrite)
    if paths.chunks_jsonl_path.exists():
        paths.chunks_jsonl_path.unlink()
    save_chunks(all_chunks, paths.chunks_jsonl_path)

    # ------------------------------------------------------------------
    # 4. Embed
    # ------------------------------------------------------------------
    embeddings = embed_chunks(all_chunks, cache_dir=paths.index_dir)

    # ------------------------------------------------------------------
    # 5. Build indices
    # ------------------------------------------------------------------
    # FAISS
    vs = VectorStore(dim=embeddings.shape[1])
    vs.build(embeddings, all_chunks)
    vs.save()

    # BM25
    bm25 = BM25Index()
    bm25.build(all_chunks)
    bm25.save()

    summary = {
        "sources_processed": len(all_sections),
        "sources_failed": len(parse_errors),
        "total_chunks": len(all_chunks),
        "embedding_dim": int(embeddings.shape[1]),
        "faiss_vectors": vs.size,
    }
    logger.info(f"Ingestion complete: {summary}")
    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RAG corpus ingestion")
    parser.add_argument("--force", action="store_true", help="Re-process all sources")
    args = parser.parse_args()
    run_ingestion(force=args.force)
