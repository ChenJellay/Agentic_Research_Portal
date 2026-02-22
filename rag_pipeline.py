"""
RAG Pipeline CLI — Phases 2 & 3 main entry point.

Subcommands
-----------
  ingest    Parse, chunk, embed, and index the corpus.
  query     Run a single RAG query with cited answer.
  evaluate  Run the full evaluation suite (20+ queries).
  acquire   Download new sources via arXiv / Semantic Scholar.
  research  Agentic pipeline: decompose → search → ingest → synthesize.

Examples::

    python rag_pipeline.py ingest
    python rag_pipeline.py query "How does AI affect code review?"
    python rag_pipeline.py evaluate
    python rag_pipeline.py acquire --search "AI software development" --max 10
    python rag_pipeline.py research "How do LLM coding agents affect testing?"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import get_model_config, get_path_config, get_rag_config
from evaluator import run_evaluation
from ingest import run_ingestion
from logger_config import RAGRunLogger, setup_logger
from manifest import get_source_by_id, load_manifest
from rag_prompts import (
    PROMPT_TEMPLATE_VERSION,
    build_rag_prompt,
    extract_citations,
    validate_citations,
)
from retriever import HybridRetriever
from source_acquisition import acquire_sources

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Query pipeline
# ---------------------------------------------------------------------------

class RAGQueryEngine:
    """
    Encapsulates the query-time RAG pipeline:
      retrieve → prompt → generate → validate citations → log.
    """

    def __init__(self) -> None:
        self.retriever = HybridRetriever()
        self.agent: Optional[Any] = None  # MLXAgent, lazily loaded
        self.manifest: List[Dict[str, Any]] = []
        self.run_logger = RAGRunLogger(get_path_config().logs_dir)
        self._chunk_id_set: set = set()

    def load(self) -> None:
        """Load retriever indices and manifest."""
        self.retriever.load()
        paths = get_path_config()
        self.manifest = load_manifest(paths.manifest_path)
        # Build set of valid chunk IDs for citation validation
        meta_path = paths.chunk_metadata_path
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            self._chunk_id_set = {m["chunk_id"] for m in meta}
        logger.info("RAG query engine loaded.")

    def _ensure_model(self):  # -> MLXAgent (lazy import)
        if self.agent is None:
            from mlx_agent import MLXAgent
            cfg = get_model_config()
            self.agent = MLXAgent(cfg)
            self.agent.initialize_model()
        return self.agent

    def query(self, question: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Execute a full RAG query.

        Returns:
            (answer_text, retrieved_chunks)
        """
        if not question.strip():
            return ("Please provide a non-empty question.", [])

        rag_cfg = get_rag_config()

        # 1. Retrieve (fill to context budget when configured)
        retrieved = self.retriever.retrieve(
            question,
            context_token_budget=rag_cfg.rag_context_budget_tokens,
        )
        logger.info(f"Retrieved {len(retrieved)} chunks for query")

        # 2. Build prompt (respect context budget)
        source_ids_in_context = {c["source_id"] for c in retrieved}
        source_meta = {}
        for sid in source_ids_in_context:
            entry = get_source_by_id(self.manifest, sid)
            if entry:
                source_meta[sid] = entry

        prompt = build_rag_prompt(
            question, retrieved, source_meta,
            context_token_budget=rag_cfg.rag_context_budget_tokens,
        )
        logger.debug(f"Prompt length: {len(prompt)} chars")

        # 3. Generate
        agent = self._ensure_model()
        answer = agent.generate_response(
            prompt,
            max_tokens=rag_cfg.rag_max_tokens,
            temperature=rag_cfg.rag_temperature,
        )

        # 4. Validate citations
        citations = extract_citations(answer)
        validation = validate_citations(citations, self._chunk_id_set)
        if validation["invalid"]:
            logger.warning(
                f"Invalid citations in answer: {validation['invalid']}"
            )

        # 5. Log
        self.run_logger.log_run(
            query=question,
            retrieved_chunks=retrieved,
            model_output=answer,
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
            model_name=get_model_config().model_name,
            citations_found=[c["chunk_id"] for c in citations],
            extra={
                "valid_citations": len(validation["valid"]),
                "invalid_citations": len(validation["invalid"]),
            },
        )

        return answer, retrieved

    def query_for_eval(self, question: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Same as query() but without model — useful for retrieval-only eval."""
        return self.query(question)


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_ingest(args: argparse.Namespace) -> None:
    """Handle the ``ingest`` subcommand."""
    logger.info("Starting ingestion pipeline…")
    summary = run_ingestion(force=args.force)
    print("\nIngestion complete:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


def cmd_query(args: argparse.Namespace) -> None:
    """Handle the ``query`` subcommand."""
    question = " ".join(args.question)
    if not question.strip():
        print("Error: please provide a non-empty query.")
        sys.exit(1)

    engine = RAGQueryEngine()
    engine.load()

    answer, retrieved = engine.query(question)

    # Display
    print("\n" + "=" * 72)
    print("QUERY:", question)
    print("=" * 72)
    print("\nRETRIEVED CHUNKS:")
    for i, c in enumerate(retrieved, 1):
        print(f"  [{i}] {c['chunk_id']}  (source: {c['source_id']}, "
              f"section: {c.get('section', 'N/A')}, score: {c.get('score', 0):.4f})")
    print("\n" + "-" * 72)
    print("ANSWER:\n")
    print(answer)
    print("\n" + "=" * 72)
    print(f"Log entry saved to: {engine.run_logger.log_path}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Handle the ``evaluate`` subcommand."""
    engine = RAGQueryEngine()
    engine.load()

    def run_query_fn(q: str) -> Tuple[str, List[Dict[str, Any]]]:
        return engine.query(q)

    # Build chunk lookup
    paths = get_path_config()
    chunk_lookup: Dict[str, str] = {}
    if paths.chunk_metadata_path.exists():
        with open(paths.chunk_metadata_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        chunk_lookup = {m["chunk_id"]: m.get("text", "") for m in meta}

    logger.info("Starting evaluation suite…")
    summary = run_evaluation(
        run_query_fn=run_query_fn,
        chunk_lookup=chunk_lookup,
    )

    print("\nEvaluation Summary:")
    print(f"  Total queries:           {summary['total_queries']}")
    print(f"  Evaluated:               {summary['evaluated']}")
    print(f"  Avg Groundedness:        {summary['avg_groundedness']:.4f}")
    print(f"  Avg Citation Precision:  {summary['avg_citation_precision']:.4f}")
    print(f"  Avg Confidence:          {summary.get('avg_confidence', 0):.4f}")
    print(f"  Failure cases:           {summary['failure_cases_count']}")
    print(f"\nDetailed results saved to: {get_path_config().eval_results_dir}")


def cmd_acquire(args: argparse.Namespace) -> None:
    """Handle the ``acquire`` subcommand."""
    queries = args.search
    max_per = args.max
    logger.info(f"Acquiring sources: queries={queries}, max_per_query={max_per}")
    added = acquire_sources(queries, max_per_query=max_per)
    print(f"\nAcquisition complete: {added} new source(s) added to manifest.")


def cmd_research(args: argparse.Namespace) -> None:
    """Handle the ``research`` subcommand (Phase 3 agentic pipeline)."""
    from config import AgentConfig
    from research_agent import ResearchAgent

    prompt = " ".join(args.prompt)
    if not prompt.strip():
        print("Error: please provide a non-empty research prompt.")
        sys.exit(1)

    config = AgentConfig(
        llm_provider=args.provider,
        agent_model=args.model,
        max_sources_per_topic=args.max_sources,
        min_year=args.min_year,
        require_peer_reviewed=args.peer_reviewed,
    )

    agent = ResearchAgent(config)
    result = agent.run(prompt)

    # Display results
    print("\n" + "=" * 72)
    print("AGENTIC RESEARCH PIPELINE — RESULTS")
    print("=" * 72)

    print("\n## Research Plan")
    for i, t in enumerate(result.plan.sub_topics, 1):
        print(f"  {i}. {t.topic_name}")
        print(f"     {t.description}")
        print(f"     Queries: {', '.join(t.search_queries)}")

    print(f"\n## Sources: {result.sources_acquired} acquired, "
          f"{result.sources_filtered} filtered out")

    if result.retrieval_stats:
        chunks = result.retrieval_stats.get("chunks_retrieved", 0)
        sids = result.retrieval_stats.get("source_ids", [])
        print(f"\n## Retrieval: {chunks} chunks from sources: {sids}")

    print("\n" + "-" * 72)
    print("SYNTHESIZED ANSWER:\n")
    print(result.answer)
    print("\n" + "=" * 72)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Research RAG Pipeline — Phases 2 & 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python rag_pipeline.py ingest
  python rag_pipeline.py ingest --force
  python rag_pipeline.py query "How does AI affect code review?"
  python rag_pipeline.py evaluate
  python rag_pipeline.py acquire --search "AI SDLC" "LLM code generation" --max 10
  python rag_pipeline.py research "How do LLM coding agents affect testing?"
  python rag_pipeline.py research "AI in code review" --provider openai --min-year 2023
""",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- ingest ---
    p_ingest = subparsers.add_parser("ingest", help="Ingest corpus: parse, chunk, embed, index")
    p_ingest.add_argument("--force", action="store_true", help="Re-process all sources")

    # --- query ---
    p_query = subparsers.add_parser("query", help="Run a single RAG query")
    p_query.add_argument("question", nargs="+", help="The question to ask")

    # --- evaluate ---
    p_eval = subparsers.add_parser("evaluate", help="Run full evaluation suite")

    # --- acquire ---
    p_acquire = subparsers.add_parser("acquire", help="Download sources from arXiv / Semantic Scholar")
    p_acquire.add_argument("--search", nargs="+", required=True, help="Search queries")
    p_acquire.add_argument("--max", type=int, default=10, help="Max results per query per API")

    # --- research (Phase 3) ---
    p_research = subparsers.add_parser(
        "research",
        help="Agentic pipeline: decompose → search → ingest → synthesize",
    )
    p_research.add_argument("prompt", nargs="+", help="Natural-language research question")
    p_research.add_argument(
        "--provider", type=str, default="mlx",
        choices=["mlx", "openai", "anthropic"],
        help="LLM provider for agent reasoning (default: mlx)",
    )
    p_research.add_argument(
        "--model", type=str, default=None,
        help="Model override for cloud providers (e.g. gpt-4o)",
    )
    p_research.add_argument(
        "--min-year", type=int, default=2022, dest="min_year",
        help="Minimum publication year (default: 2022)",
    )
    p_research.add_argument(
        "--max-sources", type=int, default=10, dest="max_sources",
        help="Max results per sub-topic query per API (default: 10)",
    )
    p_research.add_argument(
        "--peer-reviewed", action="store_true", default=True, dest="peer_reviewed",
        help="Only include peer-reviewed sources (default: True)",
    )
    p_research.add_argument(
        "--no-peer-reviewed", action="store_false", dest="peer_reviewed",
        help="Include preprints as well",
    )

    args = parser.parse_args()

    if args.debug:
        global logger
        logger = setup_logger(__name__, debug=True)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    handlers = {
        "ingest": cmd_ingest,
        "query": cmd_query,
        "evaluate": cmd_evaluate,
        "acquire": cmd_acquire,
        "research": cmd_research,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
