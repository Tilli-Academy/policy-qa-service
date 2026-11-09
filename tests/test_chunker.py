"""Tests for the chunking engine (size limits, overlap, metadata, boundaries)."""

from __future__ import annotations

from app.ingestion.chunker import _split_text, chunk_parsed_document
from app.schemas.ingestion import ParsedDocument, Section


def _doc_with_section(content: str) -> ParsedDocument:
    return ParsedDocument(
        source_filename="sample_policy.md",
        document_title="Sample Leave Policy",
        policy_id="POL-TEST-001",
        sections=[Section(level=2, heading="1. Annual Leave", hierarchy=["Sample Leave Policy", "1. Annual Leave"], content=content)],
    )


def test_chunks_respect_max_size() -> None:
    long_text = "word " * 1000  # 5000 chars
    doc = _doc_with_section(long_text)
    chunks = chunk_parsed_document(doc, max_chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(c.content) <= 200 for c in chunks)


def test_overlap_is_applied_between_chunks() -> None:
    # No whitespace -> cuts happen exactly at max_size, making overlap exact.
    text = "A" * 1000
    pieces = _split_text(text, max_size=300, overlap=50)
    assert len(pieces) > 1
    # The first 50 chars of chunk[1] equal the last 50 chars of chunk[0].
    assert pieces[1][:50] == pieces[0][-50:]


def test_short_section_is_single_chunk_without_overlap() -> None:
    doc = _doc_with_section("Short section body.")
    chunks = chunk_parsed_document(doc, max_chunk_size=800, chunk_overlap=100)
    assert len(chunks) == 1
    assert chunks[0].content == "Short section body."


def test_no_empty_chunks_produced() -> None:
    text = "Line one.\n\n\n\n   \n\n" + ("x" * 500) + "\n\n\n   \n"
    doc = _doc_with_section(text)
    chunks = chunk_parsed_document(doc, max_chunk_size=100, chunk_overlap=10)
    assert chunks
    assert all(c.content.strip() for c in chunks)


def test_sections_are_not_merged_across_boundaries() -> None:
    doc = ParsedDocument(
        source_filename="d.md",
        document_title="T",
        policy_id="POL-X-1",
        sections=[
            Section(level=2, heading="A", hierarchy=["T", "A"], content="Alpha content."),
            Section(level=2, heading="B", hierarchy=["T", "B"], content="Beta content."),
        ],
    )
    chunks = chunk_parsed_document(doc, max_chunk_size=800, chunk_overlap=100)
    assert len(chunks) == 2
    assert chunks[0].section_heading == "A" and "Alpha" in chunks[0].content
    assert chunks[1].section_heading == "B" and "Beta" in chunks[1].content
    # No chunk contains content from two sections.
    assert not any("Alpha" in c.content and "Beta" in c.content for c in chunks)


def test_metadata_is_preserved_and_index_is_sequential() -> None:
    doc = _doc_with_section("word " * 500)
    chunks = chunk_parsed_document(doc, max_chunk_size=150, chunk_overlap=20)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert chunk.source_filename == "sample_policy.md"
        assert chunk.policy_id == "POL-TEST-001"
        assert chunk.document_title == "Sample Leave Policy"
        assert chunk.section_heading == "1. Annual Leave"
        assert chunk.section_hierarchy == ["Sample Leave Policy", "1. Annual Leave"]
        assert chunk.content


def test_split_text_empty_returns_no_chunks() -> None:
    assert _split_text("   \n\n  ", max_size=100, overlap=10) == []
