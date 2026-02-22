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

## Running with MLX and Qwen (for third-party users)

The portal uses **MLX** (Apple’s machine learning framework) and **mlx-lm** to run a Qwen model locally for answer generation. If you’re trying this repo for the first time, follow these steps.

### 1. Install MLX and mlx-lm

MLX runs only on **macOS with Apple Silicon** (M1/M2/M3/M4). Install the Python packages:

```bash
pip install mlx mlx-lm
# or use the project’s full deps:
pip install -r requirements.txt
```

### 2. Choose a Qwen model

The default model is **Qwen2.5-7B-Instruct-4bit** from the MLX community:

- **Model ID**: `mlx-community/Qwen2.5-7B-Instruct-4bit`
- **Download**: About 4–5 GB; it is downloaded automatically the first time you run a query.

To use a different Qwen (or other compatible) model, set it in `config.py`:

```python
# In config.py, inside the Config class or ModelConfig:
MODEL_CONFIG = ModelConfig(
    model_name="mlx-community/Qwen2.5-7B-Instruct-4bit",  # default
    # Examples of alternatives:
    # model_name="mlx-community/Qwen2.5-3B-Instruct-4bit",   # smaller, faster
    # model_name="mlx-community/Qwen2.5-14B-Instruct-4bit-Q4", # larger
    model_path=None,  # or a local path like "/path/to/mlx_model"
    max_tokens=2048,
    temperature=0.7,
    top_p=0.9,
)
```

For RAG queries the pipeline uses lower temperature (0.3) for more faithful answers; the above is the general default.

### 3. First run: model download

On the first `query` or `evaluate` run, mlx-lm will download the model from Hugging Face into the MLX cache (usually `~/.cache/huggingface/hub/` or the path set by `HF_HOME`). No extra download script is needed.

```bash
# This will download the model on first run, then run a query
python rag_pipeline.py query "How does AI affect code review?"
```

### 4. Gated or private models

If you use a gated Hugging Face model (or a private repo), log in first:

```bash
pip install huggingface_hub
huggingface-cli login
# Enter your HF token when prompted.
```

You can instead set the token in the environment: `export HF_TOKEN=your_token_here`.

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
| `make test` | Run Phase 3 test suite (24 tests) |
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

## Phase 3 — Personal Research Portal (PRP)

Phase 3 adds a web UI and research workflow: question → evidence → synthesis → export.

### Running the PRP Locally

1. **Install dependencies** (including frontend):
   ```bash
   make install
   cd frontend && npm install
   ```
   For PDF export: `pip install weasyprint markdown fpdf2`. WeasyPrint needs system libs on macOS (`brew install pango`); if unavailable, fpdf2 is used as a fallback (pure Python).

2. **Ingest the corpus** (if not already done):
   ```bash
   make ingest
   ```

3. **Start the backend** (terminal 1):
   ```bash
   make serve
   ```
   Backend runs at http://localhost:8000. API docs at http://localhost:8000/docs.

4. **Start the frontend** (terminal 2):
   ```bash
   make dev
   ```
   Frontend runs at http://localhost:5173.

5. Open http://localhost:5173 in your browser.

### PRP Features

- **Ask**: Search bar, run RAG queries, view answers with inline citations
- **Sources**: Expandable list of retrieved chunks with metadata
- **History**: Past research threads (query + evidence + answer)
- **Artifacts**: Generate evidence table, annotated bibliography, or synthesis memo from a thread
- **Export**: Download artifacts as Markdown, CSV, or PDF
- **Evaluation**: Run the 22-query suite, view metrics and failure cases

### Trust Behavior

- Every answer includes inline citations `[source_id, chunk_id]`
- When evidence is insufficient, the system states "No sufficient evidence found in the corpus" and suggests 1–2 alternative search queries

### Phase 3 Directory Layout

```
  app/                    FastAPI backend
  frontend/               React (Vite) frontend
  data/threads/           Research thread JSON files
  artifacts/              Generated artifact outputs (for report)
  thread_store.py         File-based thread persistence
  artifact_generator.py   Evidence table, annotated bib, synthesis memo
```

## Phase 2 Deliverables

- [x] Code + repo with README instructions
- [x] `manifest.json` with metadata for every source
- [x] Evaluation framework (22-query set, 2 metrics, Markdown report)
- [x] Run logs in `logs/` (JSONL, machine-readable)
- [x] Single command (`make run-all`) for end-to-end execution

## License

[Add your license here]
