from fastapi import APIRouter
from pathlib import Path
import json

from config import get_path_config
from rag_pipeline import RAGQueryEngine
from evaluator import run_evaluation

router = APIRouter(prefix="/api", tags=["evaluation"])

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = RAGQueryEngine()
        _engine.load()
    return _engine


@router.post("/evaluation/run")
def run_eval():
    """Run the full evaluation suite."""
    engine = _get_engine()

    def run_query_fn(q):
        return engine.query(q)

    paths = get_path_config()
    chunk_lookup = {}
    if paths.chunk_metadata_path.exists():
        with open(paths.chunk_metadata_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        chunk_lookup = {m["chunk_id"]: m.get("text", "") for m in meta}

    summary = run_evaluation(
        run_query_fn=run_query_fn,
        chunk_lookup=chunk_lookup,
    )

    return {"summary": summary}


@router.get("/evaluation/latest")
def get_latest_eval():
    """Return latest evaluation results."""
    results_dir = get_path_config().eval_results_dir
    if not results_dir.exists():
        return {"summary": None, "per_query": []}

    json_files = sorted(results_dir.glob("eval_*.json"), reverse=True)
    if not json_files:
        return {"summary": None, "per_query": []}

    with open(json_files[0], "r", encoding="utf-8") as fh:
        data = json.load(fh)

    summary = data.get("summary", {})
    return {
        "summary": summary,
        "per_query": data.get("per_query", []),
    }
