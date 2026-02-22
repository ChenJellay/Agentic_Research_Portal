"""
Shared token counting for RAG pipeline.

Uses tiktoken (cl100k_base) for consistency with chunker and
transformer model tokenizers.
"""

from typing import Optional

# Lazy-loaded to avoid import cost
_enc: Optional[object] = None


def count_tokens(text: str) -> int:
    """
    Return the number of tokens for the given text using cl100k_base.

    Use this for context budgeting in retrieval and prompt construction.
    """
    import tiktoken
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding("cl100k_base")
    return len(_enc.encode(text))
