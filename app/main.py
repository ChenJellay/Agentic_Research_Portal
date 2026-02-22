"""
Phase 3 Personal Research Portal — FastAPI backend.

Run with: uvicorn app.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import query, threads, artifacts, export, evaluation

app = FastAPI(
    title="Personal Research Portal",
    description="Phase 3 — Research-grade RAG with artifacts and export",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router)
app.include_router(threads.router)
app.include_router(artifacts.router)
app.include_router(export.router)
app.include_router(evaluation.router)


@app.get("/")
def root():
    return {"message": "Personal Research Portal API", "docs": "/docs"}
