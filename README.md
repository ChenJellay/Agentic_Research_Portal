# AI Research Portal: Phase 2 — Research-Grade RAG

A research-grade Retrieval-Augmented Generation (RAG) pipeline built on a curated corpus of academic papers about AI-augmented software development.  Phase 2 adds chunking, vector + BM25 hybrid retrieval, structured citations, an evaluation framework, and production logging on top of the Phase 1 infrastructure.

## Research Domain

**Domain**: Future of work — specifically, software development.

**Main Research Question**: What does the "AI-augmented software development lifecycle / AI-gile" look like with coding agents being the main producer of code?  How do software teams adapt processes towards shipping production-level software?

## Architecture

```
rag_pipeline.py            # Main CLI entry point
├── config.py              # Configuration (model, RAG, paths, logging)
├── logger_config.py       # Console + structured JSONL logging
├── validators.py          # Input validation
├── manifest.py            # Data-manifest CRUD and validation
├── source_acquisition.py  # arXiv / Semantic Scholar paper downloader
├── pdf_processor.py       # Section-aware PDF text extraction
├── chunker.py             # Section-aware chunking with overlap
├── embedder.py            # Sentence-transformer embeddings
├── vector_store.py        # FAISS index wrapper
├── retriever.py           # Hybrid BM25 + vector retrieval (RRF)
├── rag_prompts.py         # Prompt templates with citation instructions
├── evaluator.py           # Groundedness + citation-precision metrics
├── ingest.py              # End-to-end ingestion orchestrator
└── mlx_agent.py           # MLX model interface (generation)
```

### Data Flow

```
Ingestion:   manifest.json → data/raw/ (PDFs)
             → pdf_processor (section-aware parse)
             → data/processed/ (JSON per source)
             → chunker → data/chunks/chunks.jsonl
             → embedder → vector_store + BM25 → data/index/

Query:       User question
             → retriever (hybrid BM25 + FAISS + RRF fusion)
             → rag_prompts (build prompt with context + citation rules)
             → mlx_agent (generate answer)
             → citation validation
             → logs/rag_runs.jsonl

Evaluation:  eval/queries.json (22 queries)
             → RAG pipeline → evaluator (groundedness + citation precision)
             → eval/results/ (JSON + Markdown report)
```

## Directory Layout

```
AI_Research/
  data/
    raw/              Raw PDFs (corpus artifacts)
    processed/        Extracted structured JSON per source
    chunks/           chunks.jsonl (all chunks with metadata)
    index/            FAISS index + BM25 pickle + chunk metadata
  logs/               Structured JSONL run logs
  eval/
    queries.json      22 evaluation queries
    results/          Per-run evaluation reports
  manifest.json       Source metadata manifest
  Makefile            One-command run paths
  requirements.txt    Pinned dependencies
```

## Prerequisites

- Python 3.9+
- macOS with Apple Silicon (MLX requirement)
- ~5 GB disk for model + dependencies

## Quick Start

```bash
# 1. Install dependencies
make install
# or: pip install -r requirements.txt

# 2. (Optional) Acquire additional sources from arXiv / Semantic Scholar
make acquire

# 3. Ingest corpus (parse → chunk → embed → index)
make ingest

# 4. Run a query
make query Q="How does AI affect code review in software teams?"

# 5. Run the full evaluation suite
make evaluate

# 6. Or run everything end-to-end
make run-all
```

## Commands

| Command | Description |
|---------|-------------|
| `python rag_pipeline.py ingest` | Parse, chunk, embed, and index the corpus |
| `python rag_pipeline.py ingest --force` | Re-process all sources from scratch |
| `python rag_pipeline.py query "your question"` | Single RAG query with cited answer |
| `python rag_pipeline.py evaluate` | Run 22-query evaluation suite |
| `python rag_pipeline.py acquire --search "AI SDLC" --max 10` | Download papers |
| `python rag_pipeline.py --debug <command>` | Enable verbose debug logging |

## Corpus

### Requirements Met

- **15–30 sources minimum** (4 manual + automated acquisition to reach target)
- **At least 8 peer-reviewed** papers, standards, or technical reports
- **Full metadata** for every source in `manifest.json`:
  `source_id`, `title`, `authors`, `year`, `type`, `venue`, `link/DOI`, `relevance_note`
- **Raw artifacts** stored in `data/raw/`
- **Reproducible acquisition** via `source_acquisition.py`

### manifest.json Schema

```json
{
  "source_id": "arxiv_2409_18048",
  "title": "Augmenting SE with AI...",
  "authors": ["Author A", "Author B"],
  "year": 2024,
  "type": "preprint",
  "venue": "arXiv",
  "link": "https://arxiv.org/abs/2409.18048",
  "doi": "10.48550/arXiv.2409.18048",
  "relevance_note": "Covers AI-assisted model-driven SE approach.",
  "filename": "arxiv_2409_18048.pdf",
  "acquisition_method": "manual"
}
```

## RAG Pipeline Details

### Chunking Strategy

- **Section-aware**: sections are detected via font-size heuristics and heading-pattern matching
- **Chunk size**: 512 tokens (configurable in `config.py`)
- **Overlap**: 64 tokens between consecutive chunks
- **Splitting**: at sentence boundaries when sections exceed chunk size
- **Metadata per chunk**: `chunk_id`, `source_id`, `section`, `page_start`, `page_end`, `char_start`, `char_end`

### Embeddings

- Model: `all-MiniLM-L6-v2` (384-dim, sentence-transformers)
- L2-normalised for cosine similarity via FAISS inner product
- Cached in `data/index/embeddings.npy`

### Hybrid Retrieval (Enhancement #1)

- **BM25**: `rank_bm25.BM25Okapi` for keyword matching
- **Dense vector**: FAISS `IndexFlatIP` for semantic similarity
- **Fusion**: Reciprocal Rank Fusion (RRF, k=60)
- Default: 10 candidates per retriever → 5 final chunks

### Structured Citations (Enhancement #2)

- Model is prompted to cite inline as `[source_id, chunk_id]`
- Answer ends with a `## References` section from manifest metadata
- Trust behaviour: model refuses to invent citations; flags missing/conflicting evidence
- Post-generation validation checks that cited chunk IDs exist in the index

### Generation

- Model: `Qwen2.5-7B-Instruct-4bit` via MLX
- Temperature: 0.3 (lower for faithful RAG answers)
- Max tokens: 2048

## Production Patterns

### Logging

Every query run is logged to `logs/rag_runs.jsonl` with:
- Timestamp, query text, retrieved chunk IDs with scores
- Prompt template version, model name
- Full model output, extracted citations

### Reproducibility

- Pinned dependency versions in `requirements.txt`
- One-command run via `make run-all`
- Deterministic chunk IDs (`{source_id}_chunk_{nnnn}`)

### Trust Behaviour

- System prompt explicitly instructs: "Use ONLY information from context chunks"
- "No sufficient evidence found in the corpus" when retrieval is insufficient
- Conflicting evidence is flagged with both-sides citations
- Invalid citations are logged as warnings

## Evaluation

### Query Set (22 queries)

| Type | Count | Description |
|------|-------|-------------|
| Direct | 10 | Single-fact questions with expected source evidence |
| Synthesis / Multi-hop | 5 | Cross-source comparison and aggregation |
| Edge-case / Ambiguity | 5 | Missing evidence, contradictions, absence detection |
| Stress test | 2 | Over-broad and empty queries |

### Metrics

1. **Groundedness / Faithfulness**: fraction of answer sentences supported by at least one retrieved chunk (word-overlap heuristic, threshold=0.15)
2. **Citation Precision**: fraction of inline citations whose chunk text supports the enclosing sentence

### Reports

Evaluation produces:
- `eval/results/eval_YYYYMMDD_HHMMSS.json` — detailed per-query scores
- `eval/results/eval_report_YYYYMMDD_HHMMSS.md` — human-readable report with failure cases

## Configuration

All settings are in `config.py`:

| Class | Key Settings |
|-------|-------------|
| `ModelConfig` | `model_name`, `max_tokens`, `temperature`, `top_p` |
| `RAGConfig` | `chunk_size`, `chunk_overlap`, `embedding_model`, `top_k_*`, `rrf_k` |
| `LogConfig` | `log_dir`, `log_format` |
| `PathConfig` | All directory and file paths (auto-derived from project root) |

## Phase 2 Deliverables

- [x] Code + repo with README instructions
- [x] `manifest.json` with metadata for every source
- [x] Evaluation framework (22-query set, 2 metrics, Markdown report)
- [x] Run logs in `logs/` (JSONL, machine-readable)
- [x] Single command (`make run-all`) for end-to-end execution

## License

[Add your license here]
