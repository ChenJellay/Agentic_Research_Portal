# Phase 3 Final Report — Outline

## 1. Architecture (1–2 pages)

- System overview: RAG pipeline → FastAPI backend → React frontend
- Data flow: query → retrieval → generation → thread persistence → artifact generation → export
- Key components: RAGQueryEngine, HybridRetriever, thread_store, artifact_generator

## 2. Design Choices (1–2 pages)

- **File-based threads**: Chose JSON files over DB for simplicity and reproducibility
- **Artifact schemas**: Evidence table (Claim | Evidence | Citation | Confidence | Notes); annotated bib (claim, method, limitations, why it matters); synthesis memo (800–1200 words, inline citations)
- **Trust behavior**: Inline citations; "No sufficient evidence" + suggested next retrieval steps
- **Export formats**: Markdown (direct), CSV (tabular), PDF (weasyprint from HTML)

## 3. Evaluation (1–2 pages)

- Metrics: groundedness, citation precision (from Phase 2 evaluator)
- Query set: 22 queries (direct, synthesis, edge-case, stress-test)
- Representative examples: success cases and failure cases with analysis

## 4. Limitations (1 page)

- MLX requires Apple Silicon; no cross-platform LLM option in default setup
- Artifact generation depends on LLM parsing; may need prompt tuning
- PDF export requires weasyprint (system dependencies on some platforms)

## 5. Next Steps (1 page)

- Agentic research loop with guardrails
- Knowledge graph view
- Disagreement map
- Gap finder: missing evidence + targeted retrieval
