"""
Hybrid retriever — BM25 + dense vector with Reciprocal Rank Fusion.

Enhancement #1 from the Phase 2 plan.

Pipeline
--------
1. **BM25** (``rank_bm25.BM25Okapi``): keyword-level retrieval over
   tokenised chunk texts.
2. **Dense vector** (FAISS via ``VectorStore``): semantic similarity search
   on sentence-transformer embeddings.
3. **RRF fusion**: merge both ranked lists with the Reciprocal Rank Fusion
   formula ``1 / (k + rank)`` and return the top-k final results.
"""

import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from config import get_path_config, get_rag_config
from embedder import embed_query
from logger_config import setup_logger
from token_utils import count_tokens
from vector_store import VectorStore

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Simple tokeniser for BM25
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\w+")


def _tokenise(text: str) -> List[str]:
    """Lowercase word-level tokenisation (good enough for BM25)."""
    return _WORD_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# BM25 index helpers
# ---------------------------------------------------------------------------

class BM25Index:
    """Thin wrapper around ``BM25Okapi`` with persistence."""

    def __init__(self) -> None:
        self.bm25: Optional[BM25Okapi] = None
        self.chunk_ids: List[str] = []
        self.metadata: List[Dict[str, Any]] = []

    def build(self, chunks: List[Dict[str, Any]]) -> None:
        """Build the BM25 index from chunk dicts."""
        corpus = [_tokenise(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(corpus)
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.metadata = chunks
        logger.info(f"Built BM25 index over {len(corpus)} chunk(s)")

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Return top-k BM25 results with scores."""
        if self.bm25 is None:
            return []
        tokens = _tokenise(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            entry = dict(self.metadata[idx])
            entry["score"] = float(scores[idx])
            results.append(entry)
        return results

    def save(self, path: Optional[Path] = None) -> None:
        paths = get_path_config()
        path = path or paths.bm25_index_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {"bm25": self.bm25, "chunk_ids": self.chunk_ids, "metadata": self.metadata},
                fh,
            )
        logger.info(f"Saved BM25 index → {path}")

    def load(self, path: Optional[Path] = None) -> None:
        paths = get_path_config()
        path = path or paths.bm25_index_path
        if not path.exists():
            raise FileNotFoundError(f"BM25 index not found: {path}")
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        self.bm25 = data["bm25"]
        self.chunk_ids = data["chunk_ids"]
        self.metadata = data["metadata"]
        logger.info(f"Loaded BM25 index ({len(self.chunk_ids)} chunks)")


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = 60,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Merge multiple ranked result lists using RRF.

    For each item across all lists, the RRF score is
    ``sum(1 / (k + rank_in_list_i))`` for every list it appears in.

    Args:
        ranked_lists: list of ranked result lists (each item has ``chunk_id``).
        k: RRF constant (default 60).
        top_k: number of final results.

    Returns:
        Merged and re-ranked list of chunk dicts with ``rrf_score``.
    """
    scores: Dict[str, float] = {}
    items: Dict[str, Dict[str, Any]] = {}

    for rlist in ranked_lists:
        for rank, item in enumerate(rlist, start=1):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in items:
                items[cid] = item

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_k]

    results: List[Dict[str, Any]] = []
    for cid in sorted_ids:
        entry = dict(items[cid])
        entry["rrf_score"] = scores[cid]
        # Use RRF score as the canonical score downstream
        entry["score"] = scores[cid]
        results.append(entry)

    return results


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Combines BM25 and dense-vector retrieval with RRF fusion.

    Typical usage::

        retriever = HybridRetriever()
        retriever.load()                 # loads FAISS + BM25 from disk
        results = retriever.retrieve("What does cyclomatic complexity measure?")
    """

    def __init__(self) -> None:
        self.vector_store = VectorStore()
        self.bm25_index = BM25Index()

    def load(self) -> None:
        """Load both indices from the default paths."""
        self.vector_store.load()
        self.bm25_index.load()

    def _chunk_token_count(self, chunk: Dict[str, Any]) -> int:
        """Return token count for a chunk, using metadata or estimating from text."""
        if "token_count" in chunk and isinstance(chunk["token_count"], (int, float)):
            return int(chunk["token_count"])
        return count_tokens(chunk.get("text", ""))

    def retrieve(
        self,
        query: str,
        top_k_per: Optional[int] = None,
        top_k_final: Optional[int] = None,
        context_token_budget: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run hybrid retrieval and return fused results.

        Args:
            query: natural-language question.
            top_k_per: candidates from each retriever (default from RAGConfig).
            top_k_final: final result count after RRF (default from RAGConfig).
                Ignored when context_token_budget is provided.
            context_token_budget: if provided, return chunks in RRF order that
                fit within this token budget (fill greedily).

        Returns:
            List of chunk dicts sorted by RRF score.
        """
        cfg = get_rag_config()
        top_k_per = top_k_per or cfg.top_k_per_retriever
        top_k_final = top_k_final or cfg.top_k_final

        # Dense retrieval
        qemb = embed_query(query)
        dense_results = self.vector_store.search(qemb, top_k=top_k_per)
        logger.debug(f"Dense retrieval returned {len(dense_results)} result(s)")

        # BM25 retrieval
        bm25_results = self.bm25_index.search(query, top_k=top_k_per)
        logger.debug(f"BM25 retrieval returned {len(bm25_results)} result(s)")

        # Fuse — use larger pool when filling to budget
        rrf_top_k = 50 if context_token_budget else top_k_final
        fused = reciprocal_rank_fusion(
            [dense_results, bm25_results],
            k=cfg.rrf_k,
            top_k=rrf_top_k,
        )

        if context_token_budget is not None:
            # Greedily add chunks until budget would be exceeded
            result: List[Dict[str, Any]] = []
            used = 0
            for chunk in fused:
                tc = self._chunk_token_count(chunk)
                if used + tc <= context_token_budget:
                    result.append(chunk)
                    used += tc
                else:
                    break
            logger.info(
                f"Hybrid retrieval for '{query[:60]}…': "
                f"{len(dense_results)} dense + {len(bm25_results)} BM25 → "
                f"{len(result)} chunks ({used} tokens, budget {context_token_budget})"
            )
            return result

        logger.info(
            f"Hybrid retrieval for '{query[:60]}…': "
            f"{len(dense_results)} dense + {len(bm25_results)} BM25 → {len(fused)} fused"
        )
        return fused
