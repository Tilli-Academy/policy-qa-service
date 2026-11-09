"""Groq-backed LLM implementation.

The ``groq`` SDK is imported lazily inside the constructor so merely importing
this module (e.g. during app startup or in tests) never requires the dependency
or an API key. The key is read from configuration and passed only to the SDK —
it is never logged, echoed, or included in any raised error message.
"""

from __future__ import annotations

import logging
import time

from app.config import settings
from app.llm.base import BaseLLM, LLMError, LLMTimeoutError

# Exception class names (from the groq SDK / httpx) that indicate a timeout.
# Matched by name so this module need not import the SDK at module load.
_TIMEOUT_EXC_NAMES = frozenset(
    {"APITimeoutError", "Timeout", "ReadTimeout", "ConnectTimeout", "TimeoutException"}
)

logger = logging.getLogger(__name__)


class GroqLLM(BaseLLM):
    """Chat-completion backend using Groq's OpenAI-compatible chat API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> None:
        key = api_key if api_key is not None else settings.groq_api_key
        if not key:
            # No key material in the message — just the config var name.
            raise LLMError(
                "GROQ_API_KEY is not configured. Set it in the environment / .env "
                "to enable answer generation."
            )
        self.model = model or settings.groq_model
        self._temperature = settings.groq_temperature if temperature is None else temperature
        self._max_tokens = settings.groq_max_tokens if max_tokens is None else max_tokens
        self._timeout = settings.groq_timeout_seconds if timeout is None else timeout

        try:
            from groq import Groq
        except ImportError as exc:  # pragma: no cover - exercised only without groq
            raise LLMError(
                "The 'groq' package is not installed. Install runtime dependencies "
                "with `pip install -r requirements.txt`."
            ) from exc
        self._client = Groq(api_key=key, timeout=self._timeout)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        start = time.perf_counter()
        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:  # defensive: wrap any SDK/network failure
            # Log the error TYPE and duration only — never the exception text
            # (it may contain request details) and never the key.
            duration_ms = (time.perf_counter() - start) * 1000
            if type(exc).__name__ in _TIMEOUT_EXC_NAMES:
                logger.error(
                    "Groq call timed out type=%s duration_ms=%.1f",
                    type(exc).__name__,
                    duration_ms,
                )
                raise LLMTimeoutError(
                    f"Groq request timed out ({type(exc).__name__})."
                ) from exc
            logger.error(
                "Groq call failed type=%s duration_ms=%.1f", type(exc).__name__, duration_ms
            )
            raise LLMError(f"Groq request failed ({type(exc).__name__}).") from exc

        duration_ms = (time.perf_counter() - start) * 1000
        content = completion.choices[0].message.content if completion.choices else None
        if not content:
            logger.error("Groq call returned an empty completion (duration_ms=%.1f)", duration_ms)
            raise LLMError("Groq returned an empty completion.")
        logger.info("Groq call succeeded model=%s duration_ms=%.1f", self.model, duration_ms)
        return content.strip()
