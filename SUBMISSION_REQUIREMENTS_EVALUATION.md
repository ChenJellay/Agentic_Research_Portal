# Submission Requirements Evaluation

**Date**: 2026-02-22  
**Purpose**: Evaluate the AI_Research codebase against the final submission requirements and grading rubric.

---

## Executive Summary

| Phase | Max Points | Estimated Score | Status |
|-------|------------|-----------------|--------|
| Phase 1 | 10 | ~7–8 | Improved — report/, prompt cards, framing doc added |
| Phase 2 | 15 | ~12–13 | Strong — data_manifest.json, schema compliant |
| Phase 3 | 20 | ~15–17 | Good — report folder; AI_USAGE.md template (manual fill) |
| **Total** | **45** | **~34–38** | **Address manual action items before submission** |

---

## 1. Minimum Repo Expectations

| Requirement | Status | Evidence / Gap |
|-------------|--------|----------------|
| **README with 5-min run** | ✅ Met | README has Quick Start: `make install` → `make ingest` → `make query` → `make evaluate`. Clear, sequential. |
| **Pinned dependencies** | ✅ Met | `requirements.txt` with pinned major+minor (e.g. `faiss-cpu>=1.7.4`, `sentence-transformers>=3.0.0`). No pyproject.toml; requirements.txt is acceptable. |
| **data/data_manifest.csv (or JSON)** | ✅ Met | `data/data_manifest.json` with Appendix A3 schema (`raw_path`, `processed_path`, `url_or_doi`, `tags`). |
| **data/raw/ with PDFs OR download script** | ✅ Met | 30 PDFs in `data/raw/`. `source_acquisition.py` provides reproducible acquisition from arXiv/Semantic Scholar. |
| **Outputs folder (artifacts/exports)** | ⚠️ Partial | `artifacts/` exists with sample outputs; no dedicated `outputs/` folder. Artifacts are generated on demand. |
| **Logs from evaluated runs** | ✅ Met | `logs/rag_runs.jsonl` — machine-readable JSONL with timestamp, query, retrieved_chunks, model_output, citations_found. |

---

## 2. Appendix A3 — Source Metadata Schema

**Required fields**: `source_id`, `title`, `authors`, `year`, `source_type`, `venue`, `url_or_doi`, `raw_path`, `processed_path`, `tags`, `relevance_note`

| Field | Required | In manifest.json | Notes |
|-------|----------|------------------|-------|
| source_id | ✓ | ✓ | Present |
| title | ✓ | ✓ | Present |
| authors | ✓ | ✓ | Array |
| year | ✓ | ✓ | Present |
| source_type | ✓ | `type` | Renamed; equivalent |
| venue | ✓ | ✓ | Present |
| url_or_doi | ✓ | ✓ | Present in `data/data_manifest.json` |
| raw_path | ✓ | ✓ | `data/raw/{filename}` in `data/data_manifest.json` |
| processed_path | optional | ✓ | `data/processed/{source_id}.json` in `data/data_manifest.json` |
| tags | optional | ✓ | Present (empty string if unused) |
| relevance_note | ✓ | ✓ | Present |

**Status**: Appendix A3 schema now satisfied in `data/data_manifest.json`.

---

## 3. Phase 1 (10 points)

### 3.1 Framing quality and scope discipline (3 pts)

| Criterion | Status | Evidence |
|----------|--------|----------|
| Clear question | ✅ | README: "What does the AI-augmented software development lifecycle / AI-gile look like with coding agents as main producer of code?" |
| Sub-questions | ⚠️ | Implicit in query set (direct, synthesis, edge-case) but no explicit Phase 1 framing document |
| Inclusions/exclusions | ⚠️ | Corpus scope described in README; no formal inclusions/exclusions document |

**Gap**: No dedicated Phase 1 report with explicit sub-questions and inclusion/exclusion criteria.

### 3.2 Prompt kit quality (4 pts)

| Criterion | Status | Evidence |
|----------|--------|----------|
| Structured prompts | ✅ | `rag_prompts.py`: SYSTEM_PROMPT, build_rag_prompt(), citation format, abstention rules |
| Guardrails | ✅ | "Use ONLY information from context"; "No sufficient evidence"; "Do NOT invent citations"; validate_citations() |
| Reusable prompt cards | ❌ | No prompt cards in Appendix A1 format (Prompt name, Intent, Inputs, Outputs, Constraints, When to use, Failure modes) |

**Gap**: Prompts are well-structured in code but not documented as prompt cards per template.

### 3.3 Evaluation rigor and analysis (3 pts)

| Criterion | Status | Evidence |
|----------|--------|----------|
| Consistent scoring | ✅ | `evaluator.py`: groundedness (word-overlap), citation precision (chunk-support check), confidence composite |
| Failure tags | ✅ | Failure cases in eval JSON/MD: query_id, groundedness_score, citation_precision, ungrounded_samples, answer_snippet |
| Actionable takeaways | ✅ | Eval report includes "Representative Failure Cases" with interpretation; abstention rate, answered-only metrics |

**Note**: Appendix A2 uses 1–4 scale; this evaluator uses 0–1 continuous. Functionally equivalent; could map for strict template compliance.

---

## 4. Phase 2 (15 points)

### 4.1 Corpus + manifest quality (4 pts)

| Criterion | Status | Evidence |
|----------|--------|----------|
| Metadata completeness | ✅ | 30 sources with source_id, title, authors, year, type, venue, link, doi, relevance_note, filename |
| Reproducibility | ✅ | `source_acquisition.py`, `sync_manifest.py`, `make acquire`, `make ingest` |
| Source credibility | ✅ | Mix of arXiv, ACM, journal articles; 8+ peer-reviewed (acm, ijsr, ijnrd, etc.) |

**Minor gap**: `raw_path` / `processed_path` per Appendix A3.

### 4.2 Retrieval + grounding implementation (5 pts)

| Criterion | Status | Evidence |
|----------|--------|----------|
| Working RAG | ✅ | `rag_pipeline.py`, `retriever.py`, `vector_store.py`, `embedder.py`, `chunker.py` |
| Correct citations | ✅ | `[source_id, chunk_id]` format; `validate_citations()`; post-generation validation |
| Logs | ✅ | `logs/rag_runs.jsonl` with retrieved_chunks, model_output, citations_found, valid/invalid counts |

### 4.3 Enhancement + measurable improvement (3 pts)

| Criterion | Status | Evidence |
|----------|--------|----------|
| At least one enhancement | ✅ | **Enhancement #1**: Hybrid BM25 + FAISS + RRF fusion. **Enhancement #2**: Structured citations with validation |
| Evidence it helped | ⚠️ | README describes enhancements; no before/after ablation (e.g., vector-only vs hybrid) in eval report |

**Recommendation**: Add a brief "Enhancement impact" section comparing baseline vs hybrid retrieval if such runs exist.

### 4.4 Evaluation report quality (3 pts)

| Criterion | Status | Evidence |
|----------|--------|----------|
| Query set design | ✅ | 22 queries: 10 direct, 5 synthesis, 5 edge-case, 2 stress-test; expected_behaviour per query |
| Metrics | ✅ | Groundedness, citation precision, confidence, abstention rate |
| Interpretation | ✅ | Eval report explains metrics, failure cases, abstention handling |
| Failure cases | ✅ | 5+ representative failures with query, scores, ungrounded samples, snippets |

---

## 5. Phase 3 (20 points)

### 5.1 Portal MVP functionality (8 pts)

| Criterion | Status | Evidence |
|----------|--------|----------|
| UI | ✅ | React (Vite) frontend: Ask, History, Artifacts, Evaluation pages |
| Threads | ✅ | `data/threads/`, thread_store.py, GET/POST threads API |
| Citations | ✅ | Inline `[source_id, chunk_id]` in answers; expandable chunks in UI |
| Export | ✅ | Markdown, CSV, PDF via `/api/export/{format}` |
| Reliability of core flow | ✅ | 24 tests; query → thread → artifact → export flow covered |

### 5.2 Research artifacts (4 pts)

| Criterion | Status | Evidence |
|----------|--------|----------|
| Artifact schema correctness | ✅ | Evidence table: Claim | Evidence snippet | Citation | Confidence | Notes |
| Usefulness | ✅ | Annotated bib (8–12 sources), synthesis memo (800–1200 words) |
| Citation traceability | ✅ | Citations include source_id, chunk_id; artifacts reference thread chunks |

### 5.3 Evaluation + trust behaviors (4 pts)

| Criterion | Status | Evidence |
|----------|--------|----------|
| Run query set | ✅ | `make evaluate`; `/api/evaluation/run` |
| Summarize metrics | ✅ | Eval report: aggregate + per-query tables |
| Handles missing evidence | ✅ | "No sufficient evidence found in the corpus"; "Suggested next steps: [query1], [query2]" |

### 5.4 Engineering + communication (4 pts)

| Criterion | Status | Evidence |
|----------|--------|----------|
| Clean repo | ✅ | Clear structure; Makefile; config.py |
| Run instructions | ✅ | README Quick Start, Phase 3 run steps |
| Demo | ⚠️ | Plan mentions "Demo recording (3–6 min)" — manual task; not confirmed present |
| Clear final report | ⚠️ | `artifacts/PHASE3_EVALUATION_REPORT.md`, `REPORT_OUTLINE.md`; no `report/` folder with Phase 1/2/3 writeups |

---

## 6. Recommended Repo Structure vs Actual

| Expected | Actual | Notes |
|----------|--------|-------|
| `report/` | ❌ | No report folder; Phase 3 report in `artifacts/` |
| `outputs/` | `artifacts/` | Artifacts live in `artifacts/`; exports generated on demand |
| `data/data_manifest.csv` | `manifest.json` (root) | Location + format differ |
| `src/app/`, `src/ingest/`, etc. | `app/`, root-level modules | No `src/`; functionally equivalent |

---

## 7. AI Usage Disclosure (Required)

| Requirement | Status | Evidence |
|-------------|--------|----------|
| 1-page AI-usage log | ❌ **Missing** | No `AI_USAGE.md`, `AI_DISCLOSURE.md`, or equivalent. `logs/rag_runs.jsonl` is a run log, not an AI-usage disclosure. |

**Required content**: Tool name, what you used it for, what you changed manually. Must be added before submission.

---

## 8. Action Items

### Implemented (automated)

- [x] **data/data_manifest.json**: Created with `raw_path`, `processed_path`, `url_or_doi`, `tags` (Appendix A3)
- [x] **Config**: Points to `data/data_manifest.json`; fallback to `manifest.json` if absent
- [x] **report/ folder**: `phase1_framing.md`, `phase2_report.md`, `phase3_report.md`, `README.md`
- [x] **Prompt cards**: `report/prompt_cards.md` (Appendix A1 format)
- [x] **AI_USAGE.md**: Template created; **requires manual fill-in** (see below)
- [x] **source_acquisition.py**: Adds `raw_path`, `processed_path`, `url_or_doi`, `tags` when adding sources
- [x] **README**: Updated to reference `data/data_manifest.json` and `report/`

### Manual action required

1. **Fill in AI_USAGE.md** (required): Add rows for each AI tool used — tool name, what you used it for, what you changed manually. Keep to 1 page.
2. **Demo recording (3–6 min)**: Record and link in README if not already done.
3. **Enhancement evidence** (optional): If you have baseline (vector-only) eval runs, add a before/after comparison for hybrid retrieval.

---

## 9. Strengths

- **Strong RAG implementation**: Hybrid retrieval, structured citations, validation, abstention handling.
- **Comprehensive evaluation**: 22-query set, groundedness + citation precision, failure case analysis.
- **Working portal**: Full stack (FastAPI + React), threads, artifacts, export, evaluation UI.
- **Reproducibility**: Makefile, pinned deps, acquisition script, sync script.
- **Logging**: Machine-readable JSONL for every query.
- **Test coverage**: 24 tests across API, threads, artifacts, evaluation.

---

*Evaluation complete. Address action items 1–3 for submission readiness.*
