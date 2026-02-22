# AI Research Portal — Final Submission

Research-grade RAG pipeline + Personal Research Portal for AI-augmented software development. 30-source corpus, hybrid BM25+FAISS retrieval, structured citations, 22-query evaluation, and web UI with artifacts/export.

---

## TLDR

- **What**: RAG over academic papers on AI-augmented SDLC; web portal for question → evidence → synthesis → export
- **Run**: `make install` → `make ingest` → `make query Q="..."` or `make evaluate`
- **Portal**: `make serve` + `make dev` → http://localhost:5173
- **Latest eval**: Avg groundedness 0.73, citation precision 0.78, 6 failure cases (see [Evaluation improvements](#evaluation-improvements-over-time))

---

## Domain

**Future of work** — software development.

**Main question**: What does the AI-augmented software development lifecycle look like with coding agents as the main producer of code? How do software teams adapt processes towards shipping production-level software?

---

## Architecture

```
rag_pipeline.py            # CLI entry point
├── config.py              # Paths, model, RAG settings
├── manifest.py            # data/data_manifest.json (Appendix A3)
├── source_acquisition.py  # arXiv / Semantic Scholar downloader
├── pdf_processor.py       # Section-aware PDF extraction
├── chunker.py            # 512-token chunks, 64 overlap
├── embedder.py            # all-MiniLM-L6-v2
├── vector_store.py        # FAISS index
├── retriever.py           # Hybrid BM25 + FAISS (RRF fusion)
├── rag_prompts.py         # Citation rules, abstention
├── evaluator.py           # Groundedness + citation precision
├── ingest.py              # End-to-end ingestion
└── mlx_agent.py           # Qwen2.5-7B-Instruct (MLX)
```

**Data flow**:
```
Ingestion:  data/data_manifest.json → data/raw/ (PDFs)
            → pdf_processor → data/processed/
            → chunker → data/chunks/chunks.jsonl
            → embedder + BM25 → data/index/

Query:      question → retriever → rag_prompts → mlx_agent
            → citation validation → logs/rag_runs.jsonl

Evaluation: eval/queries.json (22) → RAG → evaluator
            → eval/results/eval_YYYYMMDD_HHMMSS.{json,md}
```

---

## Runbook Commands

| Command | Description |
|---------|-------------|
| `make install` | Install Python dependencies |
| `make acquire` | Download sources (arXiv / Semantic Scholar) |
| `make ingest` | Parse, chunk, embed, index corpus |
| `make query Q="your question"` | Single RAG query |
| `make evaluate` | Run 22-query evaluation suite |
| `make run-all` | Ingest + evaluate end-to-end |
| `make serve` | Start backend (port 8000) |
| `make dev` | Start frontend (port 5173) |
| `make test` | Run test suite (24 tests) |
| `make clean` | Remove generated artifacts |

**Quick start** (5 min):
```bash
make install
make ingest
make query Q="How does AI affect code review in software teams?"
make evaluate
```

**Portal** (2 terminals):
```bash
make install && cd frontend && npm install
make ingest
make serve   # terminal 1
make dev     # terminal 2  →  http://localhost:5173
```

---

## Evaluation Improvements Over Time

Metrics from `eval/results/` show steady improvement as prompts, citation handling, and evaluator logic were refined:

| Run | Date | Avg Groundedness | Avg Citation Prec. | Avg Confidence | Failure Cases |
|-----|------|------------------|--------------------|----------------|---------------|
| [eval_report_20260215_001126](eval/results/eval_report_20260215_001126.md) | 2026-02-15 | 0.29 | 0.19 | — | 20 |
| [eval_report_20260222_005504](eval/results/eval_report_20260222_005504.md) | 2026-02-22 05:55 | 0.42 | 0.32 | 0.37 | 15 |
| [eval_report_20260222_123148](eval/results/eval_report_20260222_123148.md) | 2026-02-22 17:31 | 0.54 | 0.43 | 0.50 | 14 |
| [eval_report_20260222_125737](eval/results/eval_report_20260222_125737.md) | 2026-02-22 17:57 | **0.73** | **0.78** | **0.60** | **6** |

**Improvements**:
- **Groundedness**: 0.29 → 0.73 (2.5×)
- **Citation precision**: 0.19 → 0.78 (4×)
- **Failure cases**: 20 → 6

Key changes: citation exclusion for "Suggested next steps" and References-only lines, threshold tuning, and prompt refinements for abstention and citation format.

---

## Details

### Directory Layout

```
AI_Research/
  data/
    raw/                  Raw PDFs (30 sources)
    processed/            Parsed JSON per source
    chunks/               chunks.jsonl
    index/                FAISS + BM25 indices
    data_manifest.json    Source metadata (Appendix A3)
  eval/
    queries.json          22 evaluation queries
    results/              eval_YYYYMMDD_HHMMSS.{json,md}
  logs/                   rag_runs.jsonl (machine-readable)
  report/                 Phase 1, 2, 3 writeups, prompt cards
  app/                    FastAPI backend
  frontend/               React (Vite) frontend
  artifacts/              Sample artifact outputs
```

### Corpus

- **30 sources** (15–30 required); **8+ peer-reviewed**
- **Metadata**: `data/data_manifest.json` — source_id, title, authors, year, type, venue, link, doi, raw_path, processed_path, relevance_note
- **Acquisition**: `source_acquisition.py` (arXiv, Semantic Scholar); `make acquire`

### RAG Enhancements

1. **Hybrid retrieval**: BM25 + FAISS, RRF fusion (k=60); 10 candidates each → 5 final chunks
2. **Structured citations**: `[source_id, chunk_id]`; post-generation validation; "No sufficient evidence" + suggested next steps when abstaining

### Evaluation

- **22 queries**: 10 direct, 5 synthesis, 5 edge-case, 2 stress-test
- **Metrics**: Groundedness (word-overlap), citation precision (chunk-support check), confidence composite
- **Outputs**: `eval/results/eval_YYYYMMDD_HHMMSS.json` and `.md`

### Phase 3 — Personal Research Portal

- **Ask**: RAG queries, inline citations, expandable chunks
- **History**: Threads (query + answer + chunks)
- **Artifacts**: Evidence table, annotated bibliography, synthesis memo
- **Export**: Markdown, CSV, PDF
- **Evaluation**: Run 22-query suite from UI

### Prerequisites

- Python 3.9+
- macOS with Apple Silicon (MLX)
- ~5 GB disk (model + deps)

### Configuration

All settings in `config.py`: `ModelConfig`, `RAGConfig`, `PathConfig`, `LogConfig`.

### Deliverables Checklist

- [x] README with run instructions
- [x] `data/data_manifest.json` (Appendix A3)
- [x] `data/raw/` with PDFs + reproducible acquisition
- [x] Evaluation framework (22 queries, 2 metrics, reports)
- [x] Logs in `logs/` (JSONL)
- [x] `make run-all` for end-to-end
- [x] `report/` with Phase 1, 2, 3 writeups and prompt cards

### License

[Add your license here]
