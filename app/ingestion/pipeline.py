"""Ingestion pipeline orchestration: load -> parse -> chunk.

This is the public entry point for the ingestion milestone. It wires the three
stages together and uses the application settings for defaults so callers can
simply call :func:`ingest_documents` with no arguments.
"""

from __future__ import annotations

import logging

from app.config import settings
from app.ingestion.chunker import chunk_parsed_document
from app.ingestion.loader import load_documents
from app.ingestion.parser import parse_document
from app.schemas.ingestion import DocumentChunk

logger = logging.getLogger(__name__)


def ingest_documents(
    documents_dir: str | None = None,
    max_chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[DocumentChunk]:
    """Load, parse, and chunk every Markdown document in ``documents_dir``.

    Arguments default to the values in :data:`app.config.settings`. The returned
    list is the flattened set of chunks across all documents, each carrying the
    metadata needed for citation.
    """
    documents_dir = documents_dir if documents_dir is not None else settings.documents_dir
    max_chunk_size = max_chunk_size if max_chunk_size is not None else settings.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    chunks: list[DocumentChunk] = []
    documents = load_documents(documents_dir)
    for document in documents:
        parsed = parse_document(document)
        chunks.extend(chunk_parsed_document(parsed, max_chunk_size, chunk_overlap))

    logger.info(
        "Ingested %d document(s) from %s into %d chunk(s).",
        len(documents),
        documents_dir,
        len(chunks),
    )
    return chunks
