"""Provider-agnostic LLM interface.

Mirrors the embedder design: callers depend on :class:`BaseLLM` (an interface)
rather than a concrete provider, so the Groq backend can be swapped for a fake in
tests without any network access. The heavy / network-bound provider SDK is
imported lazily inside the concrete implementation, never here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMError(RuntimeError):
    """Raised when the LLM backend fails or is misconfigured.

    The message must never include the API key or other secrets.
    """


class LLMTimeoutError(LLMError):
    """Raised when the LLM request exceeds the configured timeout.

    A subclass of :class:`LLMError` so existing handlers still catch it, while
    callers that care can map it to a distinct response (504).
    """


class BaseLLM(ABC):
    """Abstract chat-completion backend."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return the model's completion for the given system + user prompts."""
