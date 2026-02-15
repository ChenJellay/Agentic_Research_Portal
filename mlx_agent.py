"""
MLX Agent module — Phase 2.

Interface with mlx-lm model for prompt execution and response generation.
Supports chat-template formatting for instruction-tuned models (e.g. Qwen2.5).
"""

import re
from typing import Any, List, Optional, Tuple

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

from config import ModelConfig
from logger_config import setup_logger

logger = setup_logger(__name__)


class ModelInitializationError(Exception):
    """Exception raised when model initialization fails."""
    pass


class GenerationError(Exception):
    """Exception raised when text generation fails."""
    pass


class MLXAgent:
    """Agent for interacting with MLX language models."""

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self._initialized = False

    def initialize_model(self) -> None:
        """Lazy initialization of the model."""
        if self._initialized:
            logger.debug("Model already initialized")
            return

        try:
            load_path = self.model_config.model_path or self.model_config.model_name
            logger.info(f"Loading model: {load_path}")
            logger.info("This may take a few minutes on first run...")
            self.model, self.tokenizer = load(load_path)
            self._initialized = True
            logger.info("Model loaded successfully")
        except Exception as e:
            load_path = self.model_config.model_path or self.model_config.model_name
            raise ModelInitializationError(
                f"Failed to initialize model {load_path}: {e}"
            ) from e

    def generate_response(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """
        Generate a response from the model.

        The prompt is formatted using the tokenizer's chat template when
        available (required for instruction-tuned models like Qwen2.5).
        The output is post-processed to strip stop-tokens and repetition.
        """
        if not self._initialized:
            raise ModelInitializationError(
                "Model not initialized. Call initialize_model() first."
            )

        try:
            max_tokens = max_tokens or self.model_config.max_tokens
            temperature = (
                temperature if temperature is not None else self.model_config.temperature
            )
            top_p = top_p if top_p is not None else self.model_config.top_p

            logger.debug(
                f"Generating response (max_tokens={max_tokens}, "
                f"temperature={temperature}, top_p={top_p})"
            )

            # Format with chat template
            formatted_prompt = self._format_prompt_with_context(prompt)

            sampler = make_sampler(temp=temperature, top_p=top_p)
            response = generate(
                self.model,
                self.tokenizer,
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                verbose=False,
            )

            # Post-process: strip stop tokens and repetitive tails
            response = self._post_process(response)

            logger.info(f"Generated response ({len(response)} characters)")
            return response

        except Exception as e:
            raise GenerationError(f"Failed to generate response: {e}") from e

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_prompt_with_context(self, base_prompt: str) -> str:
        """
        Wrap the prompt in the model's chat template.

        For Qwen2.5 this produces::

            <|im_start|>system
            You are a helpful assistant.<|im_end|>
            <|im_start|>user
            {base_prompt}<|im_end|>
            <|im_start|>assistant

        Falls back to raw prompt if the tokenizer has no ``apply_chat_template``.
        """
        if self.tokenizer is None:
            return base_prompt

        try:
            messages = [{"role": "user", "content": base_prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return formatted
        except Exception:
            logger.debug("Chat template not available; using raw prompt.")
            return base_prompt

    def _post_process(self, text: str) -> str:
        """
        Clean up model output:
        - Truncate at the first stop-token marker.
        - Remove repetitive tails.
        """
        # Truncate at common stop tokens
        for stop in ("<|endoftext|>", "<|im_end|>", "<|eot_id|>", "Human\n", "Human:"):
            idx = text.find(stop)
            if idx != -1:
                text = text[:idx]

        # Remove trailing whitespace
        text = text.rstrip()
        return text

    # ------------------------------------------------------------------
    # Legacy helper (kept for backward compat)
    # ------------------------------------------------------------------

    def _chunk_text_if_needed(self, text: str, max_length: int = 8192) -> List[str]:
        """Chunk text if it exceeds maximum length."""
        if len(text) <= max_length:
            return [text]
        chunks: List[str] = []
        current_chunk = ""
        for paragraph in text.split("\n\n"):
            if len(current_chunk) + len(paragraph) + 2 <= max_length:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks if chunks else [text[:max_length]]
