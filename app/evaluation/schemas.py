"""Pydantic models for the evaluation dataset.

A chunk is identified by the pair ``(source_filename, section_heading)`` — the
stable key for grading retrieval (section headings are not unique across files,
and chunk indices are brittle if the corpus is re-chunked).
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class GoldChunk(BaseModel):
    """A labeled relevant chunk, identified by file + section heading."""

    source_filename: str = Field(..., description="Policy filename, e.g. 'leave_policy.md'.")
    section_heading: str = Field(..., description="Exact section heading as indexed.")


class EvalExample(BaseModel):
    """One evaluation item: a question plus its labels."""

    id: str = Field(..., description="Unique identifier for the example.")
    question: str = Field(..., min_length=1)
    answerable: bool = Field(..., description="Whether the policies can answer this question.")
    gold_chunks: list[GoldChunk] = Field(
        default_factory=list,
        description="Relevant chunk(s). Required iff answerable; empty otherwise.",
    )
    answer_must_contain: list[str] = Field(
        default_factory=list,
        description="Case-insensitive keywords an answer must include (answer eval only).",
    )
    reference_answer: str | None = Field(
        None, description="Optional human reference answer (not auto-scored in this milestone)."
    )
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_gold_consistency(self) -> "EvalExample":
        if self.answerable and not self.gold_chunks:
            raise ValueError(f"answerable example {self.id!r} must have at least one gold chunk")
        if not self.answerable and self.gold_chunks:
            raise ValueError(f"unanswerable example {self.id!r} must not have gold chunks")
        return self
