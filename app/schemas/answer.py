"""Pydantic models for the question-answering API (Milestone 4).

These are the request/response contracts for the ``/ask`` endpoint. ``Citation``
is a thin projection of the retrieval-layer :class:`DocumentChunk` metadata, so
the API surface never leaks internal ingestion/vector details or raw chunk text
structure beyond what is needed to attribute an answer.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.config import settings
from app.schemas.retrieval import SearchResult


class Citation(BaseModel):
    """A source attribution for (part of) a generated answer."""

    source_filename: str = Field(..., description="Policy document filename, e.g. 'leave_policy.md'.")
    policy_id: str | None = Field(None, description="Policy identifier, if the document declares one.")
    document_title: str | None = Field(None, description="Human-readable document title, if known.")
    section_heading: str = Field(..., description="Heading of the cited section.")
    chunk_index: int = Field(..., description="Zero-based index of the cited chunk within its document.")
    score: float = Field(..., description="Cosine similarity of this chunk to the question; higher is more relevant.")

    @classmethod
    def from_search_result(cls, result: SearchResult) -> "Citation":
        """Project a retrieval :class:`SearchResult` into a public citation."""
        c = result.chunk
        return cls(
            source_filename=c.source_filename,
            policy_id=c.policy_id,
            document_title=c.document_title,
            section_heading=c.section_heading,
            chunk_index=c.chunk_index,
            score=result.score,
        )


class AskRequest(BaseModel):
    """A user question, optionally overriding how many chunks to retrieve."""

    question: str = Field(..., min_length=1, description="The natural-language policy question.")
    top_k: int | None = Field(
        None,
        ge=1,
        le=20,
        description="How many chunks to retrieve as context. Defaults to the configured TOP_K.",
    )

    @field_validator("question")
    @classmethod
    def _question_must_be_valid(cls, value: str) -> str:
        """Normalize whitespace and enforce non-blank + maximum length.

        ``min_length=1`` alone admits strings like ``"   "`` (non-empty but
        semantically blank). We strip, require remaining content, and cap the
        length at ``MAX_QUESTION_CHARS`` so an over-long question is rejected
        (422) rather than sent to retrieval and the LLM.
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        if len(stripped) > settings.max_question_chars:
            raise ValueError(
                f"question exceeds the maximum length of {settings.max_question_chars} characters"
            )
        return stripped


class AnswerResponse(BaseModel):
    """The generated answer plus its supporting citations."""

    question: str = Field(..., description="The question that was asked (echoed back).")
    answer: str = Field(..., description="The grounded answer, or a refusal message.")
    refused: bool = Field(
        ...,
        description="True when the policies did not contain enough information to answer.",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Sources supporting the answer. Empty when the question was refused.",
    )
