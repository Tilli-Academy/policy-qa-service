"""Offline unit tests for GroqLLM — no real Groq calls, no network.

The ``groq`` SDK client is replaced via monkeypatch (GroqLLM does
``from groq import Groq`` inside ``__init__``, so patching ``groq.Groq`` is
picked up). These tests exercise GroqLLM's own logic: API-key validation,
SDK-exception mapping, empty-completion handling, the success path, and
critically that the API key never appears in error messages or logs.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import groq
import pytest

import app.llm.groq_client as gc
from app.llm.base import LLMError
from app.llm.groq_client import GroqLLM

# A clearly-fake, non-real secret used to prove it never leaks. NOT a real key.
FAKE_SECRET = "gsk_FAKE_SECRET_DO_NOT_LEAK_0000000000000000000000"


def _install_fake_groq(monkeypatch, create_fn, record=None):
    """Replace ``groq.Groq`` with a fake whose completions.create == create_fn."""

    class _FakeGroq:
        def __init__(self, api_key=None, timeout=None, **kwargs):
            if record is not None:
                record["api_key"] = api_key
                record["timeout"] = timeout
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create_fn))

    monkeypatch.setattr(groq, "Groq", _FakeGroq)
    return _FakeGroq


def _completion(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_missing_api_key_raises_llmerror_without_leaking():
    with pytest.raises(LLMError) as exc:
        GroqLLM(api_key="")
    msg = str(exc.value)
    assert "GROQ_API_KEY" in msg          # names the config var
    assert FAKE_SECRET not in msg         # no key material


def test_missing_api_key_via_settings_fallback(monkeypatch):
    monkeypatch.setattr(gc.settings, "groq_api_key", None)
    with pytest.raises(LLMError):
        GroqLLM()  # no explicit key; settings fallback is None


def test_sdk_exception_is_mapped_and_redacted(monkeypatch):
    def boom(**kwargs):
        # Simulate a backend error whose message carelessly contains the key.
        raise RuntimeError(f"connection failed for api_key={FAKE_SECRET}")

    _install_fake_groq(monkeypatch, boom)
    llm = GroqLLM(api_key=FAKE_SECRET)
    with pytest.raises(LLMError) as exc:
        llm.generate("sys", "user")
    msg = str(exc.value)
    assert "RuntimeError" in msg          # error type surfaced for diagnostics
    assert FAKE_SECRET not in msg         # secret never in the wrapped message


def test_sdk_exception_is_not_logged_with_secret(monkeypatch, caplog):
    def boom(**kwargs):
        raise RuntimeError(f"boom {FAKE_SECRET}")

    _install_fake_groq(monkeypatch, boom)
    llm = GroqLLM(api_key=FAKE_SECRET)
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(LLMError):
            llm.generate("sys", "user")
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert FAKE_SECRET not in logged
    assert "RuntimeError" in logged       # only the type name is logged


def test_empty_choices_raises_llmerror(monkeypatch):
    _install_fake_groq(monkeypatch, lambda **k: SimpleNamespace(choices=[]))
    llm = GroqLLM(api_key=FAKE_SECRET)
    with pytest.raises(LLMError) as exc:
        llm.generate("s", "u")
    assert "empty" in str(exc.value).lower()


def test_empty_content_raises_llmerror(monkeypatch):
    _install_fake_groq(monkeypatch, lambda **k: _completion(""))
    llm = GroqLLM(api_key=FAKE_SECRET)
    with pytest.raises(LLMError):
        llm.generate("s", "u")


def test_none_content_raises_llmerror(monkeypatch):
    _install_fake_groq(monkeypatch, lambda **k: _completion(None))
    llm = GroqLLM(api_key=FAKE_SECRET)
    with pytest.raises(LLMError):
        llm.generate("s", "u")


def test_success_returns_stripped_content_and_threads_params(monkeypatch):
    record = {}
    captured = {}

    def ok(**kwargs):
        captured.update(kwargs)
        return _completion("  42 days  ")

    _install_fake_groq(monkeypatch, ok, record=record)
    llm = GroqLLM(
        api_key=FAKE_SECRET, model="test-model", temperature=0.0, max_tokens=256, timeout=12.0
    )
    out = llm.generate("SYS", "USER")

    assert out == "42 days"                                  # trimmed
    assert captured["model"] == "test-model"
    assert captured["messages"][0] == {"role": "system", "content": "SYS"}
    assert captured["messages"][1] == {"role": "user", "content": "USER"}
    # Key/timeout passed to the client constructor, never logged or returned.
    assert record["api_key"] == FAKE_SECRET
    assert record["timeout"] == 12.0
