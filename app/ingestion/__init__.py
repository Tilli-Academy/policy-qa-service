"""Markdown document ingestion pipeline (Milestone 2).

Public API:
    - ``ingest_documents``: full load -> parse -> chunk pipeline.
    - ``load_documents`` / ``discover_markdown_files``: filesystem stage.
    - ``parse_document``: Markdown -> sections stage.
    - ``chunk_parsed_document``: sections -> chunks stage.
"""

from __future__ import annotations

from app.ingestion.chunker import chunk_parsed_document
from app.ingestion.loader import discover_markdown_files, load_documents
from app.ingestion.parser import parse_document
from app.ingestion.pipeline import ingest_documents

__all__ = [
    "discover_markdown_files",
    "load_documents",
    "parse_document",
    "chunk_parsed_document",
    "ingest_documents",
]
