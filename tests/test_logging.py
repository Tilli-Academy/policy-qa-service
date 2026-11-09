"""Tests for the centralized logging system.

Offline: no Groq, no network, no real index. Verifies request-id generation and
propagation, authentication-failure logging, Groq success/failure logging, and
that secrets never appear in logs.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import groq
import pytest
from starlette.testclient import TestClient

from app import config
from app.llm.base import LLMError
from app.llm.groq_client import GroqLLM
from app.logging_config import RequestIdFilter, configure_logging, new_request_id, request_id_var
from app.main import app

FAKE_SECRET = "gsk_FAKE_LOG_SECRET_DO_NOT_LEAK"


# -- request id -----------------------------------------------------------
def test_request_id_filter_injects_contextvar():
    record = logging.LogRecord("app.x", logging.INFO, __file__, 1, "msg", None, None)
    token = request_id_var.set("abc123")
    try:
        RequestIdFilter().filter(record)
        assert record.request_id == "abc123"
    finally:
        request_id_var.reset(token)
    # Default when no request is active.
    record2 = logging.LogRecord("app.x", logging.INFO, __file__, 1, "msg", None, None)
    RequestIdFilter().filter(record2)
    assert record2.request_id == "-"


def test_new_request_id_is_unique():
    assert new_request_id() != new_request_id()


def test_request_id_header_present_and_unique():
    client = TestClient(app)
    r1 = client.get("/health")
    r2 = client.get("/health")
    assert "x-request-id" in r1.headers
    assert r1.headers["x-request-id"] != r2.headers["x-request-id"]


def test_incoming_request_id_is_honored():
    client = TestClient(app)
    r = client.get("/health", headers={"X-Request-ID": "trace-123"})
    assert r.headers["x-request-id"] == "trace-123"


# -- authentication failure logging --------------------------------------
def test_auth_failure_is_logged_without_secret(caplog, monkeypatch):
    monkeypatch.setattr(config.settings, "api_key", "SECRET-TEST-KEY")
    client = TestClient(app)
    with caplog.at_level(logging.WARNING):
        r = client.post("/ask", json={"question": "super secret question text"})
    assert r.status_code == 401
    text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "Authentication failed" in text
    assert "SECRET-TEST-KEY" not in text              # expected key never logged
    assert "super secret question text" not in text   # question never logged


# -- Groq success/failure logging ----------------------------------------
def _install_fake_groq(monkeypatch, create_fn):
    class _FakeGroq:
        def __init__(self, api_key=None, timeout=None, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create_fn))

    monkeypatch.setattr(groq, "Groq", _FakeGroq)


def test_groq_failure_is_logged_without_secret(caplog, monkeypatch):
    def boom(**kwargs):
        raise RuntimeError(f"network failure api_key={FAKE_SECRET}")

    _install_fake_groq(monkeypatch, boom)
    llm = GroqLLM(api_key=FAKE_SECRET)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(LLMError):
            llm.generate("sys", "user")
    text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "Groq call failed" in text
    assert "RuntimeError" in text        # diagnostic type present
    assert "duration_ms" in text
    assert FAKE_SECRET not in text        # secret never logged


def test_groq_success_logs_duration(caplog, monkeypatch):
    def ok(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))])

    _install_fake_groq(monkeypatch, ok)
    llm = GroqLLM(api_key="k")
    with caplog.at_level(logging.INFO):
        out = llm.generate("s", "u")
    assert out == "hi"
    text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "Groq call succeeded" in text
    assert "duration_ms" in text


# -- configuration --------------------------------------------------------
def test_configure_logging_is_idempotent():
    configure_logging()
    before = len(logging.getLogger().handlers)
    configure_logging()
    assert len(logging.getLogger().handlers) == before
