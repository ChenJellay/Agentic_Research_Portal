# Phase 3 — Personal Research Portal (PRP)

## Summary

Phase 3 adds a web UI and research workflow: question → evidence → synthesis → export.

## Portal MVP

- **UI**: React (Vite) frontend with Ask, History, Artifacts, Evaluation pages
- **Threads**: Query + answer + chunks persisted in `data/threads/`
- **Citations**: Inline `[source_id, chunk_id]` in answers; expandable chunks in UI
- **Export**: Markdown, CSV, PDF via `/api/export/{format}`
- **Evaluation**: Run 22-query suite from UI; view metrics and failure cases

## Research Artifacts

- **Evidence table**: Claim | Evidence snippet | Citation | Confidence | Notes
- **Annotated bibliography**: 8–12 sources with claim, method, limitations, why_it_matters
- **Synthesis memo**: 800–1200 words, inline citations, References section

All artifacts trace citations to source_id and chunk_id.

## Trust Behaviors

- "No sufficient evidence found in the corpus" when retrieval is insufficient
- Suggested next steps (search phrases) when abstaining
- Invalid citations logged; conflicting evidence flagged

## Run Instructions

```bash
make install && cd frontend && npm install
make ingest
make serve   # terminal 1: backend at :8000
make dev     # terminal 2: frontend at :5173
```

## Detailed Evaluation

See [artifacts/PHASE3_EVALUATION_REPORT.md](../artifacts/PHASE3_EVALUATION_REPORT.md) for component evaluation, test coverage, and gaps.
