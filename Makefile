# ============================================================================
# AI Research Portal — Phase 2 (RAG)   Makefile
# ============================================================================
# One-command run paths for reproducibility.
#
# Usage:
#   make install        Install Python dependencies
#   make acquire        Download sources from arXiv / Semantic Scholar
#   make ingest         Parse, chunk, embed, and index the corpus
#   make query Q="..."  Run a single RAG query
#   make evaluate       Run the full evaluation suite (20+ queries)
#   make run-all        Ingest + evaluate end-to-end
#   make clean          Remove generated data artifacts
# ============================================================================

PYTHON ?= python

.PHONY: install acquire ingest query evaluate run-all clean help

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install Python dependencies
	pip install -r requirements.txt

acquire:  ## Download sources (arXiv + Semantic Scholar)
	$(PYTHON) rag_pipeline.py acquire \
		--search "AI augmented software development lifecycle" \
		         "LLM code generation software engineering" \
		         "AI assisted SDLC" \
		--max 10

ingest:  ## Ingest corpus: parse → chunk → embed → index
	$(PYTHON) rag_pipeline.py ingest

ingest-force:  ## Re-ingest from scratch (ignores cache)
	$(PYTHON) rag_pipeline.py ingest --force

query:  ## Run a single query  (usage: make query Q="your question")
ifndef Q
	$(error Q is not set. Usage: make query Q="your question here")
endif
	$(PYTHON) rag_pipeline.py query "$(Q)"

evaluate:  ## Run evaluation suite (22 queries, metrics + report)
	$(PYTHON) rag_pipeline.py evaluate

run-all: ingest evaluate  ## End-to-end: ingest then evaluate

clean:  ## Remove generated data artifacts (keeps raw PDFs + manifest)
	rm -rf data/processed data/chunks data/index
	rm -rf logs/*.jsonl
	rm -rf eval/results/*
	@echo "Cleaned generated artifacts. Raw PDFs and manifest preserved."
