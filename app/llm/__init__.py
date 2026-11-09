"""LLM answer-generation package (Milestone 4).

Public API:
    - ``BaseLLM`` / ``LLMError``: the provider-agnostic interface and error type.
    - ``GroqLLM``: the Groq-backed implementation (lazy ``groq`` import).
    - ``build_messages`` / ``REFUSAL_SENTINEL``: prompt construction helpers.
"""

from __future__ import annotations

from app.llm.base import BaseLLM, LLMError
from app.llm.groq_client import GroqLLM
from app.llm.prompt import REFUSAL_SENTINEL, build_messages

__all__ = [
    "BaseLLM",
    "LLMError",
    "GroqLLM",
    "build_messages",
    "REFUSAL_SENTINEL",
]
