"""Tests for abuse-protection safeguards: rate limiting, request-size limits,
output-token forwarding, and Groq timeout handling.

Offline: real auth + real rate-limit/body-size dependencies are exercised; RAG
backends are fakes. No Groq calls, no network, no real index.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import groq
import pytest
from starlette.testclient import TestClient

from app import config
from app.api.limits import RateLimiter
from app.api.routes import get_llm, get_retriever
from app.llm.base import LLMTimeoutError
from app.llm.groq_client import GroqLLM
from app.main import app
from tests.conftest import FakeLLM, StubRetriever, make_search_result

TEST_KEY = "abuse-tests-api-key"
HEADERS = {"X-API-Key": TEST_KEY}


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def _configure(monkeypatch):
    # Real auth + real rate-limit/body-size dependencies; fake RAG backends.
    monkeypatch.setattr(config.settings, "api_key", TEST_KEY)
    app.dependency_overrides[get_retriever] = lambda: StubRetriever([make_search_result(score=0.9)])
    app.dependency_overrides[get_llm] = lambda: FakeLLM(
        "You get 25 days (leave_policy.md — 3. Annual Leave Entitlement)."
    )
    # A generous limiter by default so unrelated tests here are not throttled;
    # rate-limit tests install their own small limiter.
    app.state.rate_limiter = RateLimiter(1000, 60)
    yield
    app.dependency_overrides.clear()
    app.state.rate_limiter = None


# -- rate limiting --------------------------------------------------------
def test_requests_within_limit_succeed(client):
    app.state.rate_limiter = RateLimiter(3, 60)
    for _ in range(3):
        r = client.post("/ask", json={"question": "annual leave?"}, headers=HEADERS)
        assert r.status_code == 200


def test_requests_exceeding_limit_get_429(client):
    app.state.rate_limiter = RateLimiter(2, 60)
    assert client.post("/ask", json={"question": "q"}, headers=HEADERS).status_code == 200
    assert client.post("/ask", json={"question": "q"}, headers=HEADERS).status_code == 200
    r = client.post("/ask", json={"question": "q"}, headers=HEADERS)
    assert r.status_code == 429
    assert "Retry-After" in r.headers


def test_rate_limit_resets_after_window(client):
    clock = [1000.0]
    app.state.rate_limiter = RateLimiter(1, 60, now=lambda: clock[0])
    assert client.post("/ask", json={"question": "q"}, headers=HEADERS).status_code == 200
    assert client.post("/ask", json={"question": "q"}, headers=HEADERS).status_code == 429
    clock[0] += 61  # window elapses
    assert client.post("/ask", json={"question": "q"}, headers=HEADERS).status_code == 200


def test_health_is_not_rate_limited(client):
    app.state.rate_limiter = RateLimiter(1, 60)
    for _ in range(5):
        assert client.get("/health").status_code == 200


def test_auth_failure_does_not_consume_rate_limit(client):
    app.state.rate_limiter = RateLimiter(2, 60)
    # Invalid key -> 401, before the rate-limit dependency runs.
    for _ in range(5):
        assert client.post("/ask", json={"question": "q"}, headers={"X-API-Key": "bad"}).status_code == 401
    # Full quota is still available to the valid key.
    assert client.post("/ask", json={"question": "q"}, headers=HEADERS).status_code == 200
    assert client.post("/ask", json={"question": "q"}, headers=HEADERS).status_code == 200


def test_rate_limit_exceeded_is_logged_without_key(client, caplog):
    app.state.rate_limiter = RateLimiter(1, 60)
    with caplog.at_level(logging.WARNING):
        client.post("/ask", json={"question": "q"}, headers=HEADERS)
        client.post("/ask", json={"question": "q"}, headers=HEADERS)  # 429
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "Rate limit exceeded" in text
    assert TEST_KEY not in text


def test_rate_limiter_unit_reset():
    clock = [0.0]
    rl = RateLimiter(2, 60, now=lambda: clock[0])
    assert rl.allow("k") and rl.allow("k")
    assert not rl.allow("k")       # 3rd blocked
    clock[0] += 61
    assert rl.allow("k")           # window reset


# -- request-size limits --------------------------------------------------
def test_oversized_body_gets_413(client, monkeypatch):
    monkeypatch.setattr(config.settings, "max_request_bytes", 1024)
    big = "A" * 5000  # extra field; body well over 1 KB
    r = client.post("/ask", json={"question": "q", "filler": big}, headers=HEADERS)
    assert r.status_code == 413


def test_excessively_long_question_gets_422(client, monkeypatch):
    monkeypatch.setattr(config.settings, "max_question_chars", 50)
    monkeypatch.setattr(config.settings, "max_request_bytes", 16384)  # body stays small
    r = client.post("/ask", json={"question": "x" * 200}, headers=HEADERS)
    assert r.status_code == 422


def test_normal_request_still_works(client):
    r = client.post("/ask", json={"question": "How many annual leave days?"}, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["refused"] is False
    assert body["citations"][0]["source_filename"] == "leave_policy.md"


# -- Groq timeout handling ------------------------------------------------
def test_route_returns_504_on_llm_timeout(client):
    class TimingOutLLM:
        def generate(self, system_prompt, user_prompt):
            raise LLMTimeoutError("Groq request timed out (APITimeoutError).")

    app.dependency_overrides[get_llm] = lambda: TimingOutLLM()
    r = client.post("/ask", json={"question": "annual leave?"}, headers=HEADERS)
    assert r.status_code == 504
    assert "timed out" in r.json()["detail"].lower()


def test_groq_client_maps_timeout_and_logs_without_secret(caplog, monkeypatch):
    SECRET = "gsk_FAKE_TIMEOUT_SECRET"

    # Simulate the Groq SDK's timeout exception, matched by class name.
    class APITimeoutError(Exception):
        pass

    def boom_timeout(**kwargs):
        raise APITimeoutError(f"timed out key={SECRET}")

    class _FakeGroq:
        def __init__(self, api_key=None, timeout=None, **kw):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=boom_timeout))

    monkeypatch.setattr(groq, "Groq", _FakeGroq)
    llm = GroqLLM(api_key=SECRET)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(LLMTimeoutError):
            llm.generate("sys", "user")
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "timed out" in text.lower()
    assert SECRET not in text


# -- output-token forwarding ---------------------------------------------
def test_max_tokens_forwarded_to_groq(monkeypatch):
    captured = {}

    def ok(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))])

    class _FakeGroq:
        def __init__(self, api_key=None, timeout=None, **kw):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=ok))

    monkeypatch.setattr(groq, "Groq", _FakeGroq)
    llm = GroqLLM(api_key="k", max_tokens=321)
    llm.generate("s", "u")
    assert captured["max_tokens"] == 321  # correct supported parameter forwarded
