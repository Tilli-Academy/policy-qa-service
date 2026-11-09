"""Tests for X-API-Key authentication on the policy Q&A endpoints.

Offline: a known test key is injected via settings; RAG dependencies use fakes.
The real API_KEY is never read. TestClient runs in-process (no lifespan, no live
server, no Groq).
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from app import config
from app.api.routes import get_llm, get_retriever
from app.main import app
from tests.conftest import FakeLLM, StubRetriever, make_search_result

TEST_KEY = "unit-test-api-key-value"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def _configure(monkeypatch):
    # Configure a known server-side key and working RAG deps for /ask. Auth stays
    # REAL (these tests verify it); bypass only rate limiting and body-size.
    from app.api.limits import enforce_body_size, enforce_rate_limit

    monkeypatch.setattr(config.settings, "api_key", TEST_KEY)
    app.dependency_overrides[get_retriever] = lambda: StubRetriever([make_search_result(score=0.9)])
    app.dependency_overrides[get_llm] = lambda: FakeLLM(
        "You get 25 days (leave_policy.md — 3. Annual Leave Entitlement)."
    )
    app.dependency_overrides[enforce_rate_limit] = lambda: None
    app.dependency_overrides[enforce_body_size] = lambda: None
    yield
    app.dependency_overrides.clear()


def test_health_is_public_without_key(client):
    assert client.get("/health").status_code == 200


def test_missing_api_key_is_401(client):
    r = client.post("/ask", json={"question": "annual leave?"})
    assert r.status_code == 401
    assert TEST_KEY not in r.text  # never leak the key


def test_invalid_api_key_is_401(client):
    r = client.post("/ask", json={"question": "annual leave?"}, headers={"X-API-Key": "wrong-key"})
    assert r.status_code == 401
    assert TEST_KEY not in r.text


def test_valid_api_key_allows_rag_answer(client):
    r = client.post(
        "/ask", json={"question": "annual leave?"}, headers={"X-API-Key": TEST_KEY}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["refused"] is False
    assert "25 days" in body["answer"]
    assert body["citations"][0]["source_filename"] == "leave_policy.md"


def test_unconfigured_key_fails_closed(client, monkeypatch):
    # No server-side key => reject (503), never silently allow (not 200).
    monkeypatch.setattr(config.settings, "api_key", None)
    r = client.post(
        "/ask", json={"question": "annual leave?"}, headers={"X-API-Key": "anything"}
    )
    assert r.status_code == 503
    assert r.status_code != 200


def test_openapi_documents_x_api_key_scheme(client):
    schema = app.openapi()
    schemes = schema.get("components", {}).get("securitySchemes", {})
    assert any(
        s.get("type") == "apiKey" and s.get("in") == "header" and s.get("name") == "X-API-Key"
        for s in schemes.values()
    ), "OpenAPI should document the X-API-Key apiKey security scheme"
