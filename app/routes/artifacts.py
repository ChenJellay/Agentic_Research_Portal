from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from thread_store import load_thread
from artifact_generator import generate_evidence_table, generate_annotated_bib, generate_synthesis_memo

router = APIRouter(prefix="/api", tags=["artifacts"])


class ArtifactRequest(BaseModel):
    thread_id: str


@router.post("/artifacts/evidence-table")
def create_evidence_table(req: ArtifactRequest):
    """Generate evidence table from thread."""
    thread = load_thread(req.thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    return generate_evidence_table(thread)


@router.post("/artifacts/annotated-bib")
def create_annotated_bib(req: ArtifactRequest):
    """Generate annotated bibliography from thread."""
    thread = load_thread(req.thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    return generate_annotated_bib(thread)


@router.post("/artifacts/synthesis-memo")
def create_synthesis_memo(req: ArtifactRequest):
    """Generate synthesis memo from thread."""
    thread = load_thread(req.thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    return generate_synthesis_memo(thread)
