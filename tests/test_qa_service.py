"""Tests for the answer service: happy path, both refusal paths, error handling.

Fully offline: a FakeEmbedder-backed VectorStore provides retrieval and a FakeLLM
stands in for Groq. No FastEmbed, no network, no API key.
"""

from __future__ import annotations

import pytest

from app.llm.prompt import REFUSAL_SENTINEL
from app.retrieval.retriever import PolicyRetriever
from app.retrieval.vector_store import VectorStore
from app.schemas.ingestion import DocumentChunk
from app.services.qa import REFUSAL_MESSAGE, answer_question
from tests.conftest import FakeEmbedder, FakeLLM, RaisingLLM

CHUNKS = [
    DocumentChunk(
        source_filename="leave_policy.md",
        policy_id="POL-LEAVE-001",
        document_title="Leave Policy",
        section_heading="Annual Leave",
        section_hierarchy=["Annual Leave"],
        chunk_index=0,
        content="Full-time employees receive 25 days of paid annual leave per year.",
    ),
    DocumentChunk(
        source_filename="it_security_policy.md",
        policy_id="POL-SEC-001",
        document_title="IT Security Policy",
        section_heading="Passwords",
        section_hierarchy=["Passwords"],
        chunk_index=0,
        content="Passwords must use multi-factor authentication and rotate every 90 days.",
    ),
]


def _retriever() -> PolicyRetriever:
    store = VectorStore(FakeEmbedder(dim=64)).build(CHUNKS)
    return PolicyRetriever(store)


def test_happy_path_returns_answer_and_citations() -> None:
    llm = FakeLLM(response="You receive 25 days of annual leave (leave_policy.md — Annual Leave).")
    resp = answer_question(
        "annual leave paid days per year",
        retriever=_retriever(),
        llm=llm,
        min_score=0.0,
    )
    assert resp.refused is False
    assert "25 days" in resp.answer
    assert llm.calls == 1
    assert resp.citations
    assert resp.citations[0].source_filename == "leave_policy.md"
    assert resp.citations[0].section_heading == "Annual Leave"


def test_low_relevance_refuses_without_calling_llm() -> None:
    llm = FakeLLM()
    # A question with no shared vocabulary scores ~0, below a high floor.
    resp = answer_question(
        "zxqw unrelated gibberish token",
        retriever=_retriever(),
        llm=llm,
        min_score=0.5,
    )
    assert resp.refused is True
    assert resp.answer == REFUSAL_MESSAGE
    assert resp.citations == []
    assert llm.calls == 0  # short-circuited before the LLM


def test_model_sentinel_refuses_and_drops_citations() -> None:
    llm = FakeLLM(response=REFUSAL_SENTINEL)
    resp = answer_question(
        "annual leave paid days per year",
        retriever=_retriever(),
        llm=llm,
        min_score=0.0,
    )
    assert llm.calls == 1  # LLM was consulted
    assert resp.refused is True
    assert resp.answer == REFUSAL_MESSAGE
    assert resp.citations == []


def test_llm_error_propagates() -> None:
    from app.llm.base import LLMError

    with pytest.raises(LLMError):
        answer_question(
            "annual leave paid days per year",
            retriever=_retriever(),
            llm=RaisingLLM(),
            min_score=0.0,
        )
