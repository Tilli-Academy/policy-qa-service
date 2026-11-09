"""Unit tests for deterministic prompt construction (no LLM, no network)."""

from __future__ import annotations

from app.llm.prompt import REFUSAL_SENTINEL, SYSTEM_PROMPT, build_messages
from app.schemas.ingestion import DocumentChunk
from app.schemas.retrieval import SearchResult


def _result(filename: str, heading: str, content: str, score: float, idx: int = 0) -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(
            source_filename=filename,
            policy_id="POL-1",
            document_title="Leave Policy",
            section_heading=heading,
            section_hierarchy=[heading],
            chunk_index=idx,
            content=content,
        ),
        score=score,
    )


def test_build_messages_includes_context_and_question() -> None:
    results = [
        _result("leave_policy.md", "Annual Leave", "25 days of paid annual leave.", 0.9, 0),
        _result("leave_policy.md", "Sick Leave", "Medical certificate from day 3.", 0.5, 1),
    ]
    system, user = build_messages("How many annual leave days?", results)

    assert system == SYSTEM_PROMPT
    # Both chunks are numbered and labeled with filename + heading.
    assert "[1] leave_policy.md — Annual Leave" in user
    assert "[2] leave_policy.md — Sick Leave" in user
    assert "25 days of paid annual leave." in user
    assert "How many annual leave days?" in user


def test_system_prompt_defines_refusal_sentinel() -> None:
    assert REFUSAL_SENTINEL in SYSTEM_PROMPT


def test_build_messages_with_no_results_is_valid() -> None:
    system, user = build_messages("anything", [])
    assert system == SYSTEM_PROMPT
    assert "no relevant policy excerpts found" in user
