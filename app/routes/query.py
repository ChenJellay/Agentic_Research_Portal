from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from rag_pipeline import RAGQueryEngine
from rag_prompts import extract_citations
from thread_store import save_thread
from manifest import get_source_by_id
from config import get_path_config

router = APIRouter(prefix="/api", tags=["query"])

# Lazy-loaded engine (shared across requests)
_engine = None


def _get_engine() -> RAGQueryEngine:
    global _engine
    if _engine is None:
        _engine = RAGQueryEngine()
        _engine.load()
    return _engine


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    thread_id: str
    answer: str
    retrieved_chunks: list
    citations: list
    source_metadata: dict
    suggested_queries: list | None = None


@router.post("/query", response_model=QueryResponse)
def run_query(req: QueryRequest):
    """Run RAG query and persist to thread."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    engine = _get_engine()
    answer, retrieved = engine.query(req.question)

    citations = extract_citations(answer)
    source_ids = {c["source_id"] for c in citations}
    source_ids.update(c.get("source_id") for c in retrieved)
    source_metadata = {}
    for sid in source_ids:
        entry = get_source_by_id(engine.manifest, sid)
        if entry:
            source_metadata[sid] = entry

    suggested_queries = None
    if "No sufficient evidence" in answer or "insufficient evidence" in answer.lower():
        suggested_queries = _extract_suggested_queries(answer)

    thread_id = save_thread(
        query=req.question,
        retrieved_chunks=retrieved,
        answer=answer,
        citations=citations,
        source_metadata=source_metadata,
        suggested_queries=suggested_queries,
    )

    return QueryResponse(
        thread_id=thread_id,
        answer=answer,
        retrieved_chunks=retrieved,
        citations=citations,
        source_metadata=source_metadata,
        suggested_queries=suggested_queries,
    )


def _extract_suggested_queries(answer: str) -> list[str] | None:
    """Parse suggested next retrieval steps from answer."""
    import re
    patterns = [
        r"Suggested next steps?:\s*(.+?)(?:\n|$)",
        r"alternative (?:search )?quer(?:y|ies):\s*(.+?)(?:\n|$)",
        r"try (?:searching for|querying):\s*(.+?)(?:\n|$)",
    ]
    for pat in patterns:
        m = re.search(pat, answer, re.IGNORECASE | re.DOTALL)
        if m:
            text = m.group(1).strip()
            queries = [q.strip().strip('"\'') for q in re.split(r"[,;]|\n", text) if q.strip()]
            return queries[:2] if queries else None
    return None


@router.get("/search")
def search_only(query: str, top_k: int = 5):
    """Retrieval-only (no generation) for quick preview."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    engine = _get_engine()
    retrieved = engine.retriever.retrieve(query, top_k_final=top_k)
    return {"chunks": retrieved}
