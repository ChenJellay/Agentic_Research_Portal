from fastapi import APIRouter, HTTPException

from thread_store import load_thread, list_threads

router = APIRouter(prefix="/api", tags=["threads"])


@router.get("/threads")
def get_threads():
    """List all research threads."""
    return {"threads": list_threads()}


@router.get("/threads/{thread_id}")
def get_thread(thread_id: str):
    """Get full thread by ID."""
    thread = load_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread
