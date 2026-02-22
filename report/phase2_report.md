# Phase 2 — RAG Pipeline and Evaluation

## Summary

Phase 2 delivers a research-grade RAG pipeline with hybrid retrieval, structured citations, and an evaluation framework.

## Corpus and Manifest

- **30 sources** in `data/data_manifest.json` (Appendix A3 schema)
- **Metadata**: source_id, title, authors, year, source_type, venue, url_or_doi, raw_path, processed_path, tags, relevance_note
- **Raw artifacts**: `data/raw/` (PDFs)
- **Reproducibility**: `source_acquisition.py` for arXiv/Semantic Scholar; `make acquire`, `make ingest`

## Retrieval and Grounding

- **Hybrid retrieval**: BM25 + FAISS (RRF fusion, k=60)
- **Chunking**: Section-aware, 512 tokens, 64 overlap
- **Citations**: `[source_id, chunk_id]` format; post-generation validation
- **Trust**: "No sufficient evidence found in the corpus" + suggested next steps when retrieval is insufficient

## Enhancements (with evidence)

1. **Hybrid BM25 + vector retrieval**: Improves recall for both keyword and semantic queries vs. vector-only baseline.
2. **Structured citations**: Enforces traceability; invalid citations logged; References section from manifest metadata.

## Evaluation

- **Query set**: 22 queries (10 direct, 5 synthesis, 5 edge-case, 2 stress-test)
- **Metrics**: Groundedness (word-overlap heuristic), citation precision, confidence composite
- **Reports**: `eval/results/eval_YYYYMMDD_HHMMSS.json` and `.md`
- **Failure cases**: Representative failures with ungrounded samples and interpretation

## Logs

- `logs/rag_runs.jsonl`: Machine-readable per-query logs (timestamp, query, retrieved_chunks, model_output, citations_found)
