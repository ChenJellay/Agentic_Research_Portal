"""
Configuration module for AI Research Portal — Phases 2 & 3.

Centralizes all configuration constants, model settings, RAG parameters,
agent settings, and path mappings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Configuration for the MLX generation model."""
    model_name: str = "mlx-community/Qwen2.5-7B-Instruct-4bit"
    model_path: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9


# ---------------------------------------------------------------------------
# RAG configuration
# ---------------------------------------------------------------------------

@dataclass
class RAGConfig:
    """Configuration for the RAG pipeline."""
    # Chunking
    chunk_size: int = 512          # target tokens per chunk
    chunk_overlap: int = 64        # overlap tokens between consecutive chunks

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Retrieval
    top_k_per_retriever: int = 10  # candidates from each retriever before fusion
    top_k_final: int = 5           # chunks after RRF fusion
    rrf_k: int = 60                # RRF constant (standard default)

    # Generation
    rag_max_tokens: int = 2048
    rag_temperature: float = 0.3   # lower for more faithful answers
    prompt_template_version: str = "v1"


# ---------------------------------------------------------------------------
# Agent configuration (Phase 3 — Agentic RAG)
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Configuration for the agentic research pipeline."""
    llm_provider: str = "mlx"              # "mlx" | "openai" | "anthropic"
    agent_model: Optional[str] = None      # Cloud model override (e.g. "gpt-4o")
    max_decomposed_topics: int = 5         # Max sub-topics from decomposition
    max_sources_per_topic: int = 10        # Max API results per sub-topic query
    min_year: int = 2022                   # Recency filter: papers >= this year
    require_peer_reviewed: bool = True     # Exclude preprints (no venue)
    agent_temperature: float = 0.4         # Lower for structured JSON output
    agent_max_tokens: int = 1024           # Max tokens for agent reasoning calls


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

@dataclass
class LogConfig:
    """Configuration for structured logging."""
    log_dir: str = "logs"
    log_format: str = "jsonl"       # machine-readable format
    console_level: str = "INFO"
    file_level: str = "DEBUG"


# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------

@dataclass
class PathConfig:
    """Configuration for all file / directory paths."""
    project_root: Path = field(default_factory=lambda: Path(__file__).parent)

    # Phase 2 data directories
    data_dir: Path = field(init=False)
    raw_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)
    chunks_dir: Path = field(init=False)
    index_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    eval_dir: Path = field(init=False)
    eval_results_dir: Path = field(init=False)

    # Legacy directories (kept for backward compat)
    sources_dir: Path = field(init=False)
    output_dir: Path = field(init=False)

    # Key files
    manifest_path: Path = field(init=False)
    chunks_jsonl_path: Path = field(init=False)
    faiss_index_path: Path = field(init=False)
    bm25_index_path: Path = field(init=False)
    chunk_metadata_path: Path = field(init=False)
    eval_queries_path: Path = field(init=False)

    def __post_init__(self) -> None:
        root = self.project_root
        self.data_dir = root / "data"
        self.raw_dir = root / "data" / "raw"
        self.processed_dir = root / "data" / "processed"
        self.chunks_dir = root / "data" / "chunks"
        self.index_dir = root / "data" / "index"
        self.logs_dir = root / "logs"
        self.eval_dir = root / "eval"
        self.eval_results_dir = root / "eval" / "results"

        # Legacy
        self.sources_dir = root / "sources"
        self.output_dir = root / "output"

        # Key files
        self.manifest_path = root / "manifest.json"
        self.chunks_jsonl_path = self.chunks_dir / "chunks.jsonl"
        self.faiss_index_path = self.index_dir / "faiss.index"
        self.bm25_index_path = self.index_dir / "bm25.pkl"
        self.chunk_metadata_path = self.index_dir / "chunk_metadata.json"
        self.eval_queries_path = self.eval_dir / "queries.json"

    def ensure_dirs(self) -> None:
        """Create all directories if they don't exist."""
        for d in [
            self.raw_dir, self.processed_dir, self.chunks_dir,
            self.index_dir, self.logs_dir, self.eval_dir,
            self.eval_results_dir, self.output_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------

_path_config: Optional[PathConfig] = None
_model_config: Optional[ModelConfig] = None
_rag_config: Optional[RAGConfig] = None
_log_config: Optional[LogConfig] = None
_agent_config: Optional[AgentConfig] = None


def get_path_config() -> PathConfig:
    global _path_config
    if _path_config is None:
        _path_config = PathConfig()
    return _path_config


def get_model_config() -> ModelConfig:
    global _model_config
    if _model_config is None:
        _model_config = ModelConfig()
    return _model_config


def get_rag_config() -> RAGConfig:
    global _rag_config
    if _rag_config is None:
        _rag_config = RAGConfig()
    return _rag_config


def get_log_config() -> LogConfig:
    global _log_config
    if _log_config is None:
        _log_config = LogConfig()
    return _log_config


def get_agent_config() -> AgentConfig:
    global _agent_config
    if _agent_config is None:
        _agent_config = AgentConfig()
    return _agent_config


# ---------------------------------------------------------------------------
# Legacy helpers (kept for backward compatibility during migration)
# ---------------------------------------------------------------------------

# All available sources mapping (Phase 1 legacy)
ALL_SOURCES: Dict[str, str] = {
    "ACM 2025": "acm_2025.pdf",
    "IJSRA 2024": "ijsr_2024.pdf",
    "arXiv 2409.18048": "arxiv_2409_18048.pdf",
    "IJNRD 2024": "ijnrd_2024.pd.pdf",
}

TASK_SOURCES: Dict[int, Dict[str, str]] = {
    1: {"ACM 2025": "acm_2025.pdf", "IJSRA 2024": "ijsr_2024.pdf"},
    2: {"arXiv 2409.18048": "arxiv_2409_18048.pdf", "IJNRD 2024": "ijnrd_2024.pd.pdf"},
}


def get_task_sources(task_id: int) -> Dict[str, str]:
    if task_id not in TASK_SOURCES:
        raise ValueError(f"Invalid task_id: {task_id}. Must be 1 or 2.")
    return TASK_SOURCES[task_id]


def get_source_identifier_from_filename(filename: str) -> str:
    for identifier, fname in ALL_SOURCES.items():
        if fname == filename:
            return identifier
    return filename.replace('.pdf', '').replace('_', ' ').title()


def get_sources_from_filenames(filenames: List[str]) -> Dict[str, str]:
    return {get_source_identifier_from_filename(f): f for f in filenames}
