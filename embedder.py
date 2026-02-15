"""
Embedding module for RAG corpus.

Uses ``sentence-transformers`` with ``all-MiniLM-L6-v2`` (384-dim) to produce
dense vector representations of text chunks and queries.

Embeddings are L2-normalised so that inner-product == cosine similarity,
which is what the FAISS ``IndexFlatIP`` expects.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from config import get_rag_config
from logger_config import setup_logger

logger = setup_logger(__name__)

# Lazy-loaded model singleton
_model: Optional[Any] = None


def _get_model() -> Any:
    """Lazy-load the sentence-transformer model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        model_name = get_rag_config().embedding_model
        logger.info(f"Loading embedding model: {model_name}")
        _model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded.")
    return _model


# ---------------------------------------------------------------------------
# Core embedding functions
# ---------------------------------------------------------------------------

def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Embed a list of texts.

    Returns:
        np.ndarray of shape ``(len(texts), dim)`` with L2-normalised vectors.
    """
    model = _get_model()
    logger.info(f"Embedding {len(texts)} text(s) (batch_size={batch_size})")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    arr = np.asarray(embeddings, dtype=np.float32)
    logger.info(f"Embeddings shape: {arr.shape}")
    return arr


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns shape ``(1, dim)``."""
    return embed_texts([query])


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def save_embeddings(embeddings: np.ndarray, path: Path) -> None:
    """Persist embeddings as a ``.npy`` file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), embeddings)
    logger.info(f"Saved embeddings ({embeddings.shape}) → {path}")


def load_embeddings(path: Path) -> Optional[np.ndarray]:
    """Load cached embeddings; returns None if file missing."""
    if not path.exists():
        return None
    arr = np.load(str(path))
    logger.info(f"Loaded embeddings ({arr.shape}) from {path}")
    return arr


def embed_chunks(
    chunks: List[Dict[str, Any]],
    cache_dir: Optional[Path] = None,
) -> np.ndarray:
    """
    Embed chunk texts with optional file-system caching.

    If ``cache_dir`` is given and a cache file for this batch exists,
    the cached embeddings are returned directly.
    """
    texts = [c["text"] for c in chunks]

    if cache_dir is not None:
        cache_path = cache_dir / "embeddings.npy"
        cached = load_embeddings(cache_path)
        if cached is not None and cached.shape[0] == len(texts):
            return cached
        embeddings = embed_texts(texts)
        save_embeddings(embeddings, cache_path)
        return embeddings

    return embed_texts(texts)
