"""Pydantic models for vector search results."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.ingestion import DocumentChunk


class SearchResult(BaseModel):
    """A single similarity-search hit: the matched chunk and its cosine score."""

    chunk: DocumentChunk = Field(..., description="The matched chunk with its full citation metadata.")
    score: float = Field(..., description="Cosine similarity in [-1, 1]; higher is more similar.")
