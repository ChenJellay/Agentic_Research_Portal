"""
Agentic Research Pipeline — Phase 3.

Given a natural-language research prompt the :class:`ResearchAgent`:
  1. Decomposes the prompt into targeted sub-topics via LLM reasoning.
  2. Generates diverse search queries for each sub-topic.
  3. Calls arXiv + Semantic Scholar APIs and applies validation filters
     (recency, peer-reviewed).
  4. Downloads PDFs, updates the manifest, and runs the ingestion pipeline.
  5. Executes a RAG synthesis query over the newly ingested corpus.

Usage (standalone)::

    python research_agent.py "How do LLM-based coding agents affect testing?"

Or via the CLI::

    python rag_pipeline.py research "How do LLM-based coding agents affect testing?"
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_prompts import (
    SOURCE_RELEVANCE_PROMPT,
    SOURCE_RELEVANCE_SYSTEM,
    SYNTHESIS_PLANNING_PROMPT,
    SYNTHESIS_PLANNING_SYSTEM,
    TOPIC_DECOMPOSITION_PROMPT,
    TOPIC_DECOMPOSITION_SYSTEM,
    extract_json,
)
from config import AgentConfig, get_agent_config, get_path_config
from llm_provider import LLMProvider, get_provider
from logger_config import setup_logger
from source_acquisition import (
    acquire_sources,
    filter_by_recency,
    filter_peer_reviewed,
    search_arxiv,
    search_semantic_scholar,
)

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SubTopic:
    """A single decomposed research sub-topic."""
    topic_name: str
    description: str
    search_queries: List[str]


@dataclass
class ResearchPlan:
    """The full decomposition plan produced by the agent."""
    original_prompt: str
    sub_topics: List[SubTopic]
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class ValidatedSource:
    """A paper that passed validation filters."""
    title: str
    authors: List[str]
    year: int
    venue: str
    link: str
    pdf_url: str
    summary: str
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    passed_recency: bool = True
    passed_peer_review: bool = True
    relevance_note: str = ""


@dataclass
class AgentResult:
    """Final output of the agentic research pipeline."""
    answer: str
    plan: ResearchPlan
    sources_acquired: int
    sources_filtered: int
    retrieval_stats: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent run logger (separate from RAG run logger)
# ---------------------------------------------------------------------------

class AgentRunLogger:
    """Append-only JSONL logger for agent pipeline runs."""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "agent_runs.jsonl"

    def log(self, record: Dict[str, Any]) -> None:
        record.setdefault(
            "timestamp", datetime.now(timezone.utc).isoformat()
        )
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Research agent
# ---------------------------------------------------------------------------

class ResearchAgent:
    """
    LLM-powered research agent that decomposes a natural-language question
    into sub-topics, acquires validated sources, and produces a RAG synthesis.
    """

    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        self.config = config or get_agent_config()
        self.provider: LLMProvider = get_provider(
            self.config.llm_provider,
            model=self.config.agent_model,
        )
        self.paths = get_path_config()
        self.run_logger = AgentRunLogger(self.paths.logs_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, prompt: str) -> AgentResult:
        """
        Execute the full agentic pipeline.

        Returns an :class:`AgentResult` with the synthesised answer,
        the research plan, and acquisition statistics.
        """
        logger.info(f"Starting agentic research for: {prompt!r}")

        # 1. Decompose
        plan = self.decompose_topics(prompt)
        logger.info(
            f"Decomposed into {len(plan.sub_topics)} sub-topic(s): "
            + ", ".join(t.topic_name for t in plan.sub_topics)
        )

        # 2. Search + validate
        all_queries, candidates, validated = self.search_and_validate(plan)

        # 3. Acquire & ingest
        acquired = self.acquire_and_ingest(all_queries)

        # 4. Synthesize via RAG
        answer, retrieval_stats = self.synthesize(prompt, plan)

        result = AgentResult(
            answer=answer,
            plan=plan,
            sources_acquired=acquired,
            sources_filtered=len(candidates) - len(validated),
            retrieval_stats=retrieval_stats,
        )

        # Log the full run
        self.run_logger.log({
            "prompt": prompt,
            "sub_topics": [
                {
                    "topic_name": t.topic_name,
                    "description": t.description,
                    "search_queries": t.search_queries,
                }
                for t in plan.sub_topics
            ],
            "total_candidates": len(candidates),
            "validated_count": len(validated),
            "sources_acquired": acquired,
        })

        return result

    # ------------------------------------------------------------------
    # Step 1: Topic decomposition
    # ------------------------------------------------------------------

    def decompose_topics(self, prompt: str) -> ResearchPlan:
        """Use the LLM to break *prompt* into sub-topics with search queries."""
        system = TOPIC_DECOMPOSITION_SYSTEM.format(
            max_topics=self.config.max_decomposed_topics,
        )
        user_prompt = TOPIC_DECOMPOSITION_PROMPT.format(question=prompt)

        raw = self.provider.generate(
            prompt=user_prompt,
            system=system,
            max_tokens=self.config.agent_max_tokens,
            temperature=self.config.agent_temperature,
        )

        parsed = extract_json(raw)
        if parsed is None:
            logger.warning("LLM did not return valid JSON; using fallback plan")
            return self._fallback_plan(prompt)

        sub_topics: List[SubTopic] = []
        for item in parsed.get("sub_topics", []):
            sub_topics.append(SubTopic(
                topic_name=item.get("topic_name", "Untitled"),
                description=item.get("description", ""),
                search_queries=item.get("search_queries", [prompt]),
            ))

        if not sub_topics:
            logger.warning("LLM returned empty sub-topics; using fallback")
            return self._fallback_plan(prompt)

        # Enforce max topics limit
        sub_topics = sub_topics[: self.config.max_decomposed_topics]

        return ResearchPlan(original_prompt=prompt, sub_topics=sub_topics)

    def _fallback_plan(self, prompt: str) -> ResearchPlan:
        """Generate a minimal plan when LLM decomposition fails."""
        return ResearchPlan(
            original_prompt=prompt,
            sub_topics=[
                SubTopic(
                    topic_name="General search",
                    description="Direct search for the original query",
                    search_queries=[prompt],
                )
            ],
        )

    # ------------------------------------------------------------------
    # Step 2: Search + validate
    # ------------------------------------------------------------------

    def search_and_validate(
        self, plan: ResearchPlan
    ) -> tuple:
        """
        Execute API searches for every sub-topic query and apply filters.

        Returns:
            ``(all_queries, raw_candidates, validated_sources)``
        """
        all_queries: List[str] = []
        for topic in plan.sub_topics:
            all_queries.extend(topic.search_queries)

        # Collect raw candidates from both APIs
        candidates: List[Dict[str, Any]] = []
        for q in all_queries:
            try:
                candidates.extend(
                    search_arxiv(q, max_results=self.config.max_sources_per_topic)
                )
            except Exception as e:
                logger.error(f"arXiv search failed for '{q}': {e}")
            time.sleep(1)

            try:
                candidates.extend(
                    search_semantic_scholar(
                        q, max_results=self.config.max_sources_per_topic
                    )
                )
            except Exception as e:
                logger.error(f"S2 search failed for '{q}': {e}")
            time.sleep(1)

        logger.info(f"Raw candidates from APIs: {len(candidates)}")

        # Apply filters
        validated = list(candidates)  # copy

        if self.config.min_year:
            validated = filter_by_recency(validated, self.config.min_year)

        if self.config.require_peer_reviewed:
            filtered = filter_peer_reviewed(validated)
            if filtered:
                validated = filtered
            else:
                logger.warning(
                    "Peer-review filter removed ALL candidates — "
                    "falling back to include preprints."
                )

        # De-duplicate by title (cheap Jaccard)
        validated = self._deduplicate(validated)

        logger.info(
            f"After filtering: {len(validated)} validated source(s) "
            f"(removed {len(candidates) - len(validated)})"
        )

        return all_queries, candidates, validated

    @staticmethod
    def _deduplicate(
        papers: List[Dict[str, Any]], threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Remove near-duplicate papers by title similarity."""
        seen_titles: List[str] = []
        unique: List[Dict[str, Any]] = []

        for p in papers:
            title = p.get("title", "")
            title_words = set(title.lower().split())
            is_dup = False
            for seen in seen_titles:
                seen_words = set(seen.lower().split())
                if not title_words or not seen_words:
                    continue
                jaccard = len(title_words & seen_words) / len(
                    title_words | seen_words
                )
                if jaccard > threshold:
                    is_dup = True
                    break
            if not is_dup:
                seen_titles.append(title)
                unique.append(p)

        removed = len(papers) - len(unique)
        if removed:
            logger.info(f"De-duplication removed {removed} paper(s)")
        return unique

    # ------------------------------------------------------------------
    # Step 3: Acquire & ingest
    # ------------------------------------------------------------------

    def acquire_and_ingest(self, search_queries: List[str]) -> int:
        """
        Download PDFs via :func:`acquire_sources` and run the full
        ingestion pipeline so new sources are searchable.
        """
        added = acquire_sources(
            search_queries=search_queries,
            max_per_query=self.config.max_sources_per_topic,
            min_year=self.config.min_year,
            require_peer_reviewed=self.config.require_peer_reviewed,
        )

        if added > 0:
            logger.info(f"Acquired {added} new source(s) — running ingestion…")
            from ingest import run_ingestion

            try:
                summary = run_ingestion(force=False)
                logger.info(f"Ingestion summary: {summary}")
            except Exception as e:
                logger.error(f"Ingestion failed: {e}")
        else:
            logger.info("No new sources acquired; skipping ingestion.")

        return added

    # ------------------------------------------------------------------
    # Step 4: RAG synthesis
    # ------------------------------------------------------------------

    def synthesize(
        self, original_prompt: str, plan: ResearchPlan
    ) -> tuple:
        """
        Run a RAG query using the existing :class:`RAGQueryEngine`.

        First asks the LLM to craft an optimal synthesis query, then
        executes it through the RAG pipeline.

        Returns:
            ``(answer_text, retrieval_stats_dict)``
        """
        # Build a synthesis query (optionally LLM-assisted)
        synthesis_query = self._build_synthesis_query(original_prompt, plan)

        # Import here to avoid circular imports
        from rag_pipeline import RAGQueryEngine

        engine = RAGQueryEngine()
        try:
            engine.load()
        except Exception as e:
            logger.error(f"Failed to load RAG engine: {e}")
            return (
                "Unable to load the RAG engine. The corpus may not have "
                "been ingested yet. Run the pipeline again or check logs.",
                {},
            )

        answer, retrieved = engine.query(synthesis_query)

        retrieval_stats = {
            "synthesis_query": synthesis_query,
            "chunks_retrieved": len(retrieved),
            "source_ids": list({c.get("source_id") for c in retrieved}),
        }

        return answer, retrieval_stats

    def _build_synthesis_query(
        self, original_prompt: str, plan: ResearchPlan
    ) -> str:
        """
        Optionally use the LLM to craft a better synthesis query.

        Falls back to the original prompt if LLM call fails.
        """
        topics_summary = "\n".join(
            f"- {t.topic_name}: {t.description}" for t in plan.sub_topics
        )

        try:
            raw = self.provider.generate(
                prompt=SYNTHESIS_PLANNING_PROMPT.format(
                    question=original_prompt,
                    topics_summary=topics_summary,
                    source_count=len(plan.sub_topics),
                ),
                system=SYNTHESIS_PLANNING_SYSTEM,
                max_tokens=512,
                temperature=self.config.agent_temperature,
            )
            parsed = extract_json(raw)
            if parsed and "synthesis_query" in parsed:
                query = parsed["synthesis_query"]
                logger.info(f"Synthesis query: {query!r}")
                return query
        except Exception as e:
            logger.warning(f"Synthesis planning LLM call failed: {e}")

        # Fallback: use original prompt
        return original_prompt


# ---------------------------------------------------------------------------
# CLI entry point (standalone)
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Agentic research pipeline — decompose, search, ingest, synthesize"
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="Natural-language research question",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="mlx",
        choices=["mlx", "openai", "anthropic"],
        help="LLM provider for agent reasoning (default: mlx)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model override for cloud providers (e.g. gpt-4o)",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=2022,
        help="Minimum publication year (default: 2022)",
    )
    parser.add_argument(
        "--max-sources",
        type=int,
        default=10,
        help="Max results per sub-topic query per API (default: 10)",
    )
    parser.add_argument(
        "--no-peer-reviewed",
        action="store_true",
        help="Include preprints (by default only peer-reviewed)",
    )
    args = parser.parse_args()

    config = AgentConfig(
        llm_provider=args.provider,
        agent_model=args.model,
        max_sources_per_topic=args.max_sources,
        min_year=args.min_year,
        require_peer_reviewed=not args.no_peer_reviewed,
    )

    agent = ResearchAgent(config)
    result = agent.run(args.prompt)

    # Display
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
        print(f"\n## Retrieval: {result.retrieval_stats.get('chunks_retrieved', 0)} "
              f"chunks from {result.retrieval_stats.get('source_ids', [])}")

    print("\n" + "-" * 72)
    print("SYNTHESIZED ANSWER:\n")
    print(result.answer)
    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
