"""End-to-end ingestion tests, including the real sample documents."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.pipeline import ingest_documents

EXPECTED_FILES = {
    "leave_policy.md",
    "remote_work_policy.md",
    "expense_reimbursement_policy.md",
    "it_security_policy.md",
    "access_control_policy.md",
}


def test_ingest_sample_fixture_dir(sample_docs_dir: Path) -> None:
    chunks = ingest_documents(str(sample_docs_dir), max_chunk_size=200, chunk_overlap=20)
    assert chunks
    sources = {c.source_filename for c in chunks}
    assert sources == {"sample_policy.md", "plain_policy.md"}
    assert all(c.content.strip() for c in chunks)
    assert all(len(c.content) <= 200 for c in chunks)


def test_ingest_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ingest_documents(str(tmp_path / "nope"))


def test_ingest_real_documents(real_docs_dir: Path) -> None:
    chunks = ingest_documents(str(real_docs_dir), max_chunk_size=800, chunk_overlap=100)
    assert chunks, "expected at least one chunk from the real documents"

    sources = {c.source_filename for c in chunks}
    assert sources == EXPECTED_FILES

    # Every real document declares a policy id and a title; every chunk must
    # carry them along with non-empty content within the size limit.
    for chunk in chunks:
        assert chunk.policy_id is not None
        assert chunk.document_title is not None
        assert chunk.section_heading
        assert chunk.section_hierarchy
        assert chunk.content.strip()
        assert len(chunk.content) <= 800


def test_real_documents_chunk_index_sequential_per_document(real_docs_dir: Path) -> None:
    chunks = ingest_documents(str(real_docs_dir), max_chunk_size=800, chunk_overlap=100)
    per_doc: dict[str, list[int]] = {}
    for chunk in chunks:
        per_doc.setdefault(chunk.source_filename, []).append(chunk.chunk_index)
    for filename, indices in per_doc.items():
        assert indices == list(range(len(indices))), f"non-sequential indices in {filename}"


def test_known_fact_is_retrievable_in_chunks(real_docs_dir: Path) -> None:
    chunks = ingest_documents(str(real_docs_dir))
    leave_chunks = [c for c in chunks if c.source_filename == "leave_policy.md"]
    assert any("25 days" in c.content for c in leave_chunks)
    assert any(c.policy_id == "POL-HR-001" for c in leave_chunks)
