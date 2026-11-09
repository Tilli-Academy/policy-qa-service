"""HTTP-level tests for the /ask route.

Uses Starlette's TestClient WITHOUT the context-manager form, so the app's
lifespan (which would load FastEmbed + the real index) never runs. The retriever
and LLM are supplied via FastAPI dependency overrides with offline fakes.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from app.api.routes import get_llm, get_retriever
from app.main import app
from app.retrieval.retriever import PolicyRetriever
from app.retrieval.vector_store import VectorStore
from app.schemas.ingestion import DocumentChunk
from tests.conftest import FakeEmbedder, FakeLLM, RaisingLLM

CHUNK = DocumentChunk(
    source_filename="leave_policy.md",
    policy_id="POL-LEAVE-001",
    document_title="Leave Policy",
    section_heading="Annual Leave",
    section_hierarchy=["Annual Leave"],
    chunk_index=0,
    content="Full-time employees receive 25 days of paid annual leave per year.",
)


def _retriever() -> PolicyRetriever:
    store = VectorStore(FakeEmbedder(dim=64)).build([CHUNK])
    return PolicyRetriever(store)


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


def test_ask_happy_path_returns_answer_and_citations(client: TestClient) -> None:
    app.dependency_overrides[get_retriever] = _retriever
    app.dependency_overrides[get_llm] = lambda: FakeLLM(
        response="You receive 25 days (leave_policy.md — Annual Leave)."
    )
    resp = client.post("/ask", json={"question": "annual leave paid days per year"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["refused"] is False
    assert "25 days" in body["answer"]
    assert body["citations"][0]["source_filename"] == "leave_policy.md"


def test_ask_refuses_on_low_relevance(client: TestClient) -> None:
    app.dependency_overrides[get_retriever] = _retriever
    app.dependency_overrides[get_llm] = lambda: FakeLLM()
    # Unrelated question scores below the default ANSWER_MIN_SCORE floor.
    resp = client.post("/ask", json={"question": "zxqw unrelated gibberish token"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["refused"] is True
    assert body["citations"] == []


def test_ask_llm_failure_returns_502(client: TestClient) -> None:
    app.dependency_overrides[get_retriever] = _retriever
    app.dependency_overrides[get_llm] = lambda: RaisingLLM()
    resp = client.post("/ask", json={"question": "annual leave paid days per year"})
    assert resp.status_code == 502


def test_ask_missing_backend_returns_503(client: TestClient) -> None:
    # No overrides and no lifespan => app.state has no retriever/llm => 503.
    resp = client.post("/ask", json={"question": "anything"})
    assert resp.status_code == 503


def test_ask_rejects_empty_question(client: TestClient) -> None:
    app.dependency_overrides[get_retriever] = _retriever
    app.dependency_overrides[get_llm] = lambda: FakeLLM()
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 422
