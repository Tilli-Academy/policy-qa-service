"""HTTP-level RAG behavior tests for /ask: error mapping, top_k passthrough and
validation bounds, whitespace rejection, empty-retrieval refusal, and secret
non-exposure.

Offline: dependency overrides supply fakes, and TestClient is used WITHOUT the
context-manager form so the app lifespan (FastEmbed + real index) never runs and
no live server/network is touched.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from app.api.routes import get_llm, get_retriever
from app.llm.base import BaseLLM, LLMError
from app.main import app
from tests.conftest import FakeLLM, RaisingRetriever, StubRetriever, make_search_result

FAKE_SECRET = "gsk_FAKE_ROUTE_SECRET_DO_NOT_LEAK_111111111111111"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides():
    # These tests exercise RAG behavior: bypass auth, rate limiting, and body-size
    # checks (each has its own dedicated test module).
    from app.api.auth import require_api_key
    from app.api.limits import enforce_body_size, enforce_rate_limit

    app.dependency_overrides[require_api_key] = lambda: None
    app.dependency_overrides[enforce_rate_limit] = lambda: None
    app.dependency_overrides[enforce_body_size] = lambda: None
    yield
    app.dependency_overrides.clear()


def _use(retriever, llm) -> None:
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_llm] = lambda: llm


def test_embedding_error_maps_to_502(client):
    _use(RaisingRetriever(), FakeLLM())
    r = client.post("/ask", json={"question": "anything"})
    assert r.status_code == 502
    assert "embedding" in r.json()["detail"].lower()
    assert FAKE_SECRET not in r.text


def test_top_k_is_passed_through_to_retriever(client):
    stub = StubRetriever([make_search_result(score=0.9)])
    _use(stub, FakeLLM("ans"))
    r = client.post("/ask", json={"question": "q", "top_k": 3})
    assert r.status_code == 200
    assert stub.last_top_k == 3


def test_top_k_defaults_to_none_when_omitted(client):
    stub = StubRetriever([make_search_result(score=0.9)])
    _use(stub, FakeLLM("ans"))
    r = client.post("/ask", json={"question": "q"})
    assert r.status_code == 200
    assert stub.last_top_k is None


@pytest.mark.parametrize("bad", [0, -1, 21, 100])
def test_top_k_out_of_bounds_is_422(client, bad):
    _use(StubRetriever([make_search_result(score=0.9)]), FakeLLM("ans"))
    r = client.post("/ask", json={"question": "q", "top_k": bad})
    assert r.status_code == 422


def test_whitespace_only_question_is_422(client):
    _use(StubRetriever([make_search_result(score=0.9)]), FakeLLM("ans"))
    r = client.post("/ask", json={"question": "   "})
    assert r.status_code == 422


def test_question_is_stripped_in_response(client):
    stub = StubRetriever([make_search_result(score=0.9)])
    _use(stub, FakeLLM("ans"))
    r = client.post("/ask", json={"question": "  How many leave days?  "})
    assert r.status_code == 200
    assert r.json()["question"] == "How many leave days?"


def test_empty_retrieval_returns_200_refused(client):
    _use(StubRetriever([]), FakeLLM())
    r = client.post("/ask", json={"question": "q"})
    assert r.status_code == 200
    body = r.json()
    assert body["refused"] is True
    assert body["citations"] == []


def test_llm_error_response_does_not_leak_secret(client):
    class LeakyLLM(BaseLLM):
        def generate(self, system_prompt, user_prompt):
            # Careless backend message containing a secret; route must not echo it.
            raise LLMError(f"auth failed key={FAKE_SECRET}")

    _use(StubRetriever([make_search_result(score=0.9)]), LeakyLLM())
    r = client.post("/ask", json={"question": "q"})
    assert r.status_code == 502
    assert FAKE_SECRET not in r.text
    assert "LLM backend error" in r.json()["detail"]
