"""Pydantic models for the document ingestion pipeline.

These models are the typed contracts passed between the ingestion stages:
loader -> parser -> chunker. ``DocumentChunk`` is the final output consumed by
later milestones (retrieval / embeddings).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LoadedDocument(BaseModel):
    """A raw Markdown file read from disk."""

    source_filename: str = Field(..., description="Base filename, e.g. 'leave_policy.md'.")
    content: str = Field(..., description="Full UTF-8 file contents.")


class Section(BaseModel):
    """A contiguous block of content under a single Markdown heading."""

    level: int = Field(..., description="Heading level (1-6); 0 if the section has no heading.")
    heading: str = Field(..., description="The immediate section heading text.")
    hierarchy: list[str] = Field(
        default_factory=list,
        description="Ancestor headings from the top level down to this heading, inclusive.",
    )
    content: str = Field(..., description="Section body text (excluding the heading line).")


class ParsedDocument(BaseModel):
    """A document after title/policy-id extraction and sectioning."""

    source_filename: str
    document_title: str | None = None
    policy_id: str | None = None
    sections: list[Section] = Field(default_factory=list)


class DocumentChunk(BaseModel):
    """A single retrievable chunk with the metadata required for citations."""

    source_filename: str
    policy_id: str | None = None
    document_title: str | None = None
    section_heading: str
    section_hierarchy: list[str] = Field(default_factory=list)
    chunk_index: int = Field(..., description="Zero-based index of this chunk within its document.")
    content: str
