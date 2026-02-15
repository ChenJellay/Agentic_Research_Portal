"""
LLM Provider abstraction — Phase 3 (Agentic RAG).

Provides a configurable interface so the research agent's reasoning can run
on either the local MLX model or a cloud LLM API (OpenAI / Anthropic).

All providers expose a single ``generate()`` method that accepts a user
prompt and an optional system prompt, returning plain text.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional

from logger_config import setup_logger

logger = setup_logger(__name__)


class LLMProviderError(Exception):
    """Raised when an LLM provider fails to generate a response."""
    pass


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """Base class for all LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.4,
    ) -> str:
        """Generate a text completion.

        Args:
            prompt: The user/instruction prompt.
            system: Optional system prompt prepended to the conversation.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            The generated text as a plain string.
        """
        ...


# ---------------------------------------------------------------------------
# MLX local provider (wraps existing MLXAgent)
# ---------------------------------------------------------------------------

class MLXProvider(LLMProvider):
    """Delegates to the local MLX model via :class:`mlx_agent.MLXAgent`."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        from config import get_model_config, ModelConfig

        if model_name:
            self._config = ModelConfig(model_name=model_name)
        else:
            self._config = get_model_config()
        self._agent = None  # lazy

    def _ensure_agent(self):
        if self._agent is None:
            from mlx_agent import MLXAgent
            self._agent = MLXAgent(self._config)
            self._agent.initialize_model()
        return self._agent

    def generate(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.4,
    ) -> str:
        agent = self._ensure_agent()
        # Prepend system prompt if provided
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        try:
            return agent.generate_response(
                full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            raise LLMProviderError(f"MLX generation failed: {e}") from e


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    """Uses the OpenAI Chat Completions API (requires ``OPENAI_API_KEY``)."""

    def __init__(self, model: str = "gpt-4o") -> None:
        self.model = model
        self._client = None  # lazy

    def _ensure_client(self):
        if self._client is None:
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise LLMProviderError(
                    "OPENAI_API_KEY not set. Export it or add to .env file."
                )

            try:
                import openai
            except ImportError:
                raise LLMProviderError(
                    "openai package not installed. Run: pip install openai"
                )

            self._client = openai.OpenAI(api_key=api_key)
        return self._client

    def generate(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.4,
    ) -> str:
        client = self._ensure_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            raise LLMProviderError(f"OpenAI generation failed: {e}") from e


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    """Uses the Anthropic Messages API (requires ``ANTHROPIC_API_KEY``)."""

    def __init__(self, model: str = "claude-sonnet-4-20250514") -> None:
        self.model = model
        self._client = None  # lazy

    def _ensure_client(self):
        if self._client is None:
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise LLMProviderError(
                    "ANTHROPIC_API_KEY not set. Export it or add to .env file."
                )

            try:
                import anthropic
            except ImportError:
                raise LLMProviderError(
                    "anthropic package not installed. Run: pip install anthropic"
                )

            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def generate(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.4,
    ) -> str:
        client = self._ensure_client()

        try:
            kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system:
                kwargs["system"] = system

            response = client.messages.create(**kwargs)
            # Anthropic returns a list of content blocks
            return "".join(
                block.text for block in response.content if block.type == "text"
            )
        except Exception as e:
            raise LLMProviderError(f"Anthropic generation failed: {e}") from e


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_provider(provider_name: str, model: Optional[str] = None) -> LLMProvider:
    """
    Create an :class:`LLMProvider` by name.

    Args:
        provider_name: One of ``"mlx"``, ``"openai"``, ``"anthropic"``.
        model: Optional model override (e.g. ``"gpt-4o-mini"``).

    Returns:
        An initialised provider instance (model loaded lazily on first call).
    """
    name = provider_name.lower().strip()

    if name == "mlx":
        return MLXProvider(model_name=model)
    elif name == "openai":
        return OpenAIProvider(model=model or "gpt-4o")
    elif name == "anthropic":
        return AnthropicProvider(model=model or "claude-sonnet-4-20250514")
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider_name!r}. "
            f"Choose from: mlx, openai, anthropic"
        )
