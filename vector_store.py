"""
FAISS vector store for RAG retrieval.

Stores L2-normalised embeddings in a ``faiss.IndexFlatIP`` (inner-product
equals cosine similarity for unit vectors).  A sidecar JSON maps integer
index positions to chunk metadata (chunk_id, source_id, etc.).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from config import get_path_config, get_rag_config
from logger_config import setup_logger

logger = setup_logger(__name__)


class VectorStore:
    """Thin wrapper around a FAISS flat inner-product index."""

    def __init__(self, dim: Optional[int] = None) -> None:
        cfg = get_rag_config()
        self.dim = dim or cfg.embedding_dim
        self.index: Optional[faiss.IndexFlatIP] = None
        # Ordered list of chunk metadata; position matches FAISS row.
        self.metadata: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        embeddings: np.ndarray,
        chunks_metadata: List[Dict[str, Any]],
    ) -> None:
        """
        Build the index from scratch.

        Args:
            embeddings: shape ``(n, dim)``, L2-normalised float32.
            chunks_metadata: list of dicts (one per row) with at least
                ``chunk_id`` and ``source_id``.
        """
        if embeddings.shape[0] != len(chunks_metadata):
            raise ValueError(
                f"Embedding rows ({embeddings.shape[0]}) != metadata length ({len(chunks_metadata)})"
            )
        self.dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings.astype(np.float32))
        self.metadata = list(chunks_metadata)
        logger.info(
            f"Built FAISS index: {self.index.ntotal} vectors, dim={self.dim}"
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self, query_embedding: np.ndarray, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Return top-k results with scores.

        Args:
            query_embedding: shape ``(1, dim)`` or ``(dim,)``.
            top_k: number of results.

        Returns:
            List of dicts with ``score`` plus all metadata fields.
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Vector index is empty; returning no results.")
            return []

        qv = query_embedding.reshape(1, -1).astype(np.float32)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(qv, k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            entry = dict(self.metadata[idx])
            entry["score"] = float(score)
            results.append(entry)
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, index_path: Optional[Path] = None, meta_path: Optional[Path] = None) -> None:
        """Save FAISS index + sidecar metadata JSON."""
        paths = get_path_config()
        index_path = index_path or paths.faiss_index_path
        meta_path = meta_path or paths.chunk_metadata_path

        index_path.parent.mkdir(parents=True, exist_ok=True)

        if self.index is not None:
            faiss.write_index(self.index, str(index_path))
            logger.info(f"Saved FAISS index → {index_path}")

        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(self.metadata, fh, ensure_ascii=False, indent=2)
        logger.info(f"Saved chunk metadata ({len(self.metadata)} entries) → {meta_path}")

    def load(self, index_path: Optional[Path] = None, meta_path: Optional[Path] = None) -> None:
        """Load FAISS index + sidecar metadata JSON."""
        paths = get_path_config()
        index_path = index_path or paths.faiss_index_path
        meta_path = meta_path or paths.chunk_metadata_path

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Chunk metadata not found: {meta_path}")

        self.index = faiss.read_index(str(index_path))
        self.dim = self.index.d
        logger.info(f"Loaded FAISS index ({self.index.ntotal} vectors, dim={self.dim})")

        with open(meta_path, "r", encoding="utf-8") as fh:
            self.metadata = json.load(fh)
        logger.info(f"Loaded chunk metadata ({len(self.metadata)} entries)")

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return self.index.ntotal if self.index else 0

    def get_chunk_text(self, chunk_id: str) -> Optional[str]:
        """Look up the raw text for a chunk_id."""
        for m in self.metadata:
            if m.get("chunk_id") == chunk_id:
                return m.get("text")
        return None
