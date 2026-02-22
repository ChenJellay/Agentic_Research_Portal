# Prompt Cards (Appendix A1 Format)

## Card 1: RAG Answer Generation

**Prompt name**: RAG System Prompt + Answer Generation

**Intent**: Generate a grounded, cited answer to a research question using only provided context chunks. Enforce abstention when evidence is insufficient.

**Inputs (what you provide)**:
- System prompt (fixed rules)
- Context chunks (retrieved from corpus) with chunk_id, source_id, section, text
- Source metadata block (title, authors, year, venue, link per source)
- User question

**Outputs (required structure)**:
- Answer body with inline citations `[source_id, chunk_id]` after each supported claim
- OR "No sufficient evidence found in the corpus." + "Suggested next steps: [search phrase 1], [search phrase 2]"
- `## References` section listing every cited source with metadata

**Constraints / guardrails**:
- No fabricated citations; every chunk_id must appear in the context
- Cite chunk IDs only; do not use `[source_id=X, chunk_id=Y]` or bare chunk_ids
- Do NOT put `[source_id, chunk_id]` in Suggested next steps — use plain search phrases
- Flag conflicting evidence and cite both sides

**When to use / when not to use**:
- Use: Single-turn RAG query over curated corpus
- Not for: Open-ended generation, web search, or when context is empty

**Failure modes to watch for**:
- Citations in "Suggested next steps" (model confusion; evaluator excludes these)
- References section lines counted as ungrounded (metadata-like text)
- Invalid chunk_ids (typos, wrong source_id); post-generation validation catches these

---

## Card 2: Topic Decomposition (Agent)

**Prompt name**: Topic Decomposition for Academic Search

**Intent**: Break a research question into 2–5 focused sub-topics, each with search query variations for arXiv/Semantic Scholar.

**Inputs (what you provide)**:
- Research question
- Max sub-topics (default 5)

**Outputs (required structure)**:
- JSON: `{"sub_topics": [{"topic_name": "...", "description": "...", "search_queries": ["q1", "q2"]}]}`

**Constraints / guardrails**:
- Output ONLY valid JSON; no markdown or commentary
- Search queries: 3–8 words, specific technical terminology
- Avoid overly broad queries

**When to use / when not to use**:
- Use: Agentic pipeline when decomposing a broad question for multi-query retrieval
- Not for: Single direct RAG query

**Failure modes to watch for**:
- Malformed JSON; `extract_json` handles with fallback
- Too few or too many sub-topics; validate against max_topics

---

## Card 3: Source Relevance Scoring (Agent)

**Prompt name**: Source Relevance Judge

**Intent**: Score (1–5) whether a paper's title/abstract is relevant to the research question.

**Inputs (what you provide)**:
- Research question
- Paper title and abstract

**Outputs (required structure)**:
- JSON: `{"relevance_score": 1-5, "justification": "..."}`

**Constraints / guardrails**:
- Be strict: score 3+ only if paper directly addresses the research area
- Output ONLY valid JSON

**When to use / when not to use**:
- Use: Agentic pipeline when filtering API search results before acquisition
- Not for: In-corpus retrieval (that uses BM25/FAISS)

**Failure modes to watch for**:
- Overly lenient scores; papers outside domain included
- JSON parse failure; fallback to low score

---

## Card 4: Evidence Table Extraction (Artifact)

**Prompt name**: Evidence Table from Thread

**Intent**: Extract factual claims from an answer and map each to supporting evidence (snippet, source_id, chunk_id, confidence).

**Inputs (what you provide)**:
- Query, answer, retrieved chunks (text)

**Outputs (required structure)**:
- JSON: `{"rows": [{"claim": "...", "evidence_snippet": "...", "source_id": "...", "chunk_id": "...", "confidence": 0.0-1.0, "notes": "..."}]}`

**Constraints / guardrails**:
- Evidence snippet: complete quote from chunk, do not truncate mid-sentence
- Citation traceability: source_id and chunk_id must be valid

**When to use / when not to use**:
- Use: Generating evidence table artifact from a research thread
- Not for: Real-time query answering

**Failure modes to watch for**:
- LLM parsing fails → fallback row "No claims extracted"
- Truncated evidence snippets; cap at 600 chars in markdown for display
