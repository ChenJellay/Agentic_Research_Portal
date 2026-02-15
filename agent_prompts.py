"""
Agent reasoning prompts — Phase 3 (Agentic RAG).

Prompt templates used by :class:`research_agent.ResearchAgent` for:
  - Topic decomposition (breaking a research question into sub-topics).
  - Source relevance scoring (LLM-judged filtering of API results).
  - Synthesis planning (generating the final RAG query).

All prompts request structured JSON output, parsed by :func:`extract_json`.
"""

import json
import re
from typing import Any, Optional

from logger_config import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Topic decomposition
# ---------------------------------------------------------------------------

TOPIC_DECOMPOSITION_SYSTEM = """\
You are a research planning assistant. Given a research question, your job is \
to decompose it into focused sub-topics that can each be searched independently \
in academic databases (arXiv, Semantic Scholar).

RULES:
1. Output ONLY valid JSON — no markdown, no commentary.
2. Produce between 2 and {max_topics} sub-topics.
3. Each sub-topic must have diverse search_queries (2-3 keyword variations) \
that are effective for academic paper search engines.
4. Keep search queries concise (3-8 words each) and specific to the sub-topic.
5. Avoid overly broad queries; prefer precise technical terminology."""

TOPIC_DECOMPOSITION_PROMPT = """\
Decompose the following research question into focused sub-topics for \
academic literature search.

Research question: {question}

Return a JSON object with this exact schema:
{{
  "sub_topics": [
    {{
      "topic_name": "Short descriptive name",
      "description": "One-sentence description of what this sub-topic covers",
      "search_queries": ["query 1", "query 2", "query 3"]
    }}
  ]
}}"""


# ---------------------------------------------------------------------------
# Source relevance scoring
# ---------------------------------------------------------------------------

SOURCE_RELEVANCE_SYSTEM = """\
You are an academic relevance judge. Given a paper's title and abstract, \
and the original research question, assess whether this paper is relevant.

RULES:
1. Output ONLY valid JSON — no markdown, no commentary.
2. Score from 1 (completely irrelevant) to 5 (highly relevant).
3. Be strict: score 3+ only if the paper directly addresses the research area."""

SOURCE_RELEVANCE_PROMPT = """\
Rate the relevance of this paper to the research question.

Research question: {question}

Paper title: {title}
Paper abstract: {abstract}

Return a JSON object:
{{
  "relevance_score": <1-5>,
  "justification": "One sentence explaining the score"
}}"""


# ---------------------------------------------------------------------------
# Synthesis planning
# ---------------------------------------------------------------------------

SYNTHESIS_PLANNING_SYSTEM = """\
You are a research synthesis planner. Given a set of decomposed sub-topics \
and the sources that have been acquired, generate the best single query to \
run against the RAG system for a comprehensive synthesis.

RULES:
1. Output ONLY valid JSON — no markdown, no commentary.
2. The query should be comprehensive enough to pull relevant chunks from \
all acquired sources.
3. Reference the key themes from the sub-topics."""

SYNTHESIS_PLANNING_PROMPT = """\
Generate a RAG query that will produce a comprehensive synthesis.

Original research question: {question}

Sub-topics investigated:
{topics_summary}

Sources acquired: {source_count} papers

Return a JSON object:
{{
  "synthesis_query": "The comprehensive query to run against the RAG system",
  "focus_areas": ["area 1", "area 2", "area 3"]
}}"""


# ---------------------------------------------------------------------------
# JSON extraction utility
# ---------------------------------------------------------------------------

def extract_json(text: str) -> Optional[Any]:
    """
    Extract a JSON object or array from LLM output.

    Handles common quirks:
      - Markdown code fences (````json ... ````).
      - Leading/trailing prose around the JSON block.
      - Trailing commas before closing braces/brackets.

    Returns the parsed Python object, or ``None`` if extraction fails.
    """
    if not text or not text.strip():
        return None

    # 1. Try to extract from markdown code fence
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1).strip()
        parsed = _try_parse(candidate)
        if parsed is not None:
            return parsed

    # 2. Try to find a top-level JSON object or array
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start_idx = text.find(start_char)
        if start_idx == -1:
            continue
        # Find the matching closing character (handle nesting)
        depth = 0
        for i in range(start_idx, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
                if depth == 0:
                    candidate = text[start_idx : i + 1]
                    parsed = _try_parse(candidate)
                    if parsed is not None:
                        return parsed
                    break

    # 3. Last resort: try the whole string
    parsed = _try_parse(text.strip())
    if parsed is not None:
        return parsed

    logger.warning(f"Failed to extract JSON from LLM output (length={len(text)})")
    return None


def _try_parse(text: str) -> Optional[Any]:
    """Attempt to parse JSON, fixing trailing commas if needed."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fix trailing commas: ,} or ,]
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None
