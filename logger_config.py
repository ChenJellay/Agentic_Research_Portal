"""
Logging configuration module — Phase 2.

Provides:
  - Console logging (human-readable) via ``setup_logger``
  - Structured JSONL file logging for RAG run records via ``RAGRunLogger``
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Console logger (carries forward from Phase 1)
# ---------------------------------------------------------------------------

def setup_logger(
    name: str = "research_agent",
    level: int = logging.INFO,
    debug: bool = False,
) -> logging.Logger:
    """
    Set up and configure a logger instance for console output.

    Args:
        name: Logger name (typically module name).
        level: Logging level (default: INFO).
        debug: If True, sets level to DEBUG with verbose formatting.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG if debug else level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if debug else level)

    if debug:
        formatter = logging.Formatter(
            "[%(levelname)s] [%(name)s] [%(funcName)s:%(lineno)d] %(message)s"
        )
    else:
        formatter = logging.Formatter("[%(levelname)s] [%(name)s] %(message)s")

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Structured JSONL run logger for RAG queries
# ---------------------------------------------------------------------------

class RAGRunLogger:
    """
    Append-only JSONL logger that persists one record per RAG query.

    Each record contains the query, retrieved chunks, model output,
    prompt/version metadata, and citation information.
    """

    def __init__(self, log_dir: str | Path = "logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.log_dir / "rag_runs.jsonl"

    @property
    def log_path(self) -> Path:
        return self._log_path

    # ------------------------------------------------------------------
    def log_run(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        model_output: str,
        prompt_template_version: str,
        model_name: str,
        citations_found: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Log a single RAG query run.

        Returns the full record dict (useful for tests / callers).
        """
        record: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "retrieved_chunks": [
                {
                    "chunk_id": c.get("chunk_id", ""),
                    "source_id": c.get("source_id", ""),
                    "score": round(c.get("score", 0.0), 6),
                }
                for c in retrieved_chunks
            ],
            "prompt_template_version": prompt_template_version,
            "model_name": model_name,
            "model_output": model_output,
            "citations_found": citations_found or [],
        }
        if extra:
            record["extra"] = extra

        with open(self._log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        return record

    # ------------------------------------------------------------------
    def read_all(self) -> List[Dict[str, Any]]:
        """Read all log records (useful for evaluation)."""
        if not self._log_path.exists():
            return []
        records: List[Dict[str, Any]] = []
        with open(self._log_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
