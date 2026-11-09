"""Tests for the document loader (discovery + reading)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.ingestion.loader import discover_markdown_files, load_documents


def test_loads_multiple_markdown_files(sample_docs_dir: Path) -> None:
    docs = load_documents(sample_docs_dir)
    names = {d.source_filename for d in docs}
    # Only the two valid, non-empty, non-hidden .md files should load.
    assert names == {"sample_policy.md", "plain_policy.md"}


def test_ignores_non_markdown_and_dotfiles(sample_docs_dir: Path) -> None:
    names = {p.name for p in discover_markdown_files(sample_docs_dir)}
    assert "notes.txt" not in names       # wrong extension
    assert ".env" not in names            # dotfile / secret
    assert ".hidden.md" not in names      # hidden markdown
    assert names == {"sample_policy.md", "plain_policy.md", "empty.md"}


def test_empty_document_is_skipped(sample_docs_dir: Path) -> None:
    docs = load_documents(sample_docs_dir)
    assert all(d.source_filename != "empty.md" for d in docs)
    assert all(d.content.strip() for d in docs)


def test_missing_directory_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        discover_markdown_files(missing)
    with pytest.raises(FileNotFoundError):
        load_documents(missing)


def test_path_that_is_not_a_directory_raises(tmp_path: Path) -> None:
    a_file = tmp_path / "afile.md"
    a_file.write_text("# x", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        discover_markdown_files(a_file)


def test_invalid_utf8_file_is_skipped(tmp_path: Path) -> None:
    docs_dir = tmp_path / "policies"
    docs_dir.mkdir()
    (docs_dir / "good.md").write_text("# Good\nvalid", encoding="utf-8")
    # Bytes that are not valid UTF-8 -> read_text raises UnicodeDecodeError.
    (docs_dir / "bad.md").write_bytes(b"\xff\xfe\x00bad bytes")

    docs = load_documents(docs_dir)
    names = {d.source_filename for d in docs}
    assert names == {"good.md"}  # bad file skipped gracefully, good file loads


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root can read files regardless of permission bits",
)
def test_unreadable_file_is_skipped(tmp_path: Path) -> None:
    docs_dir = tmp_path / "policies"
    docs_dir.mkdir()
    good = docs_dir / "good.md"
    good.write_text("# Good\nvalid", encoding="utf-8")
    bad = docs_dir / "locked.md"
    bad.write_text("# Locked\nsecret", encoding="utf-8")
    os.chmod(bad, 0o000)
    try:
        docs = load_documents(docs_dir)
    finally:
        os.chmod(bad, 0o644)  # restore so tmp cleanup works
    assert {d.source_filename for d in docs} == {"good.md"}
