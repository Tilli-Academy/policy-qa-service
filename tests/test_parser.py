"""Tests for the Markdown parser (title/policy id + sectioning + hierarchy)."""

from __future__ import annotations

from app.ingestion.parser import parse_document
from app.schemas.ingestion import LoadedDocument
from tests.conftest import NO_HEADING_MD, SAMPLE_MD


def _parse(content: str, name: str = "doc.md"):
    return parse_document(LoadedDocument(source_filename=name, content=content))


def test_extracts_title_and_policy_id() -> None:
    parsed = _parse(SAMPLE_MD)
    assert parsed.document_title == "Sample Leave Policy"
    assert parsed.policy_id == "POL-TEST-001"


def test_policy_id_is_none_when_absent() -> None:
    parsed = _parse("# Title\n\nNo identifier here.")
    assert parsed.policy_id is None


def test_headings_are_extracted_as_sections() -> None:
    parsed = _parse(SAMPLE_MD)
    headings = [s.heading for s in parsed.sections]
    assert "Sample Leave Policy" in headings
    assert "1. Annual Leave" in headings
    assert "1.1 Accrual" in headings
    assert "2. Sick Leave" in headings


def test_section_hierarchy_is_nested() -> None:
    parsed = _parse(SAMPLE_MD)
    by_heading = {s.heading: s for s in parsed.sections}

    # H2 under the H1 title.
    assert by_heading["1. Annual Leave"].hierarchy == [
        "Sample Leave Policy",
        "1. Annual Leave",
    ]
    # H3 under H2 under H1.
    assert by_heading["1.1 Accrual"].hierarchy == [
        "Sample Leave Policy",
        "1. Annual Leave",
        "1.1 Accrual",
    ]
    # A sibling H2 must not inherit the previous H2/H3 as ancestors.
    assert by_heading["2. Sick Leave"].hierarchy == [
        "Sample Leave Policy",
        "2. Sick Leave",
    ]


def test_document_without_headings_is_single_section() -> None:
    parsed = _parse(NO_HEADING_MD, name="plain_policy.md")
    assert parsed.document_title is None
    assert len(parsed.sections) == 1
    section = parsed.sections[0]
    # Falls back to the filename stem for the heading.
    assert section.heading == "plain_policy"
    assert section.hierarchy == ["plain_policy"]
    assert "preserved" in section.content  # content not discarded


def test_meaningful_content_is_preserved() -> None:
    parsed = _parse(SAMPLE_MD)
    combined = "\n".join(s.content for s in parsed.sections)
    assert "25 days of paid annual leave" in combined
    assert "medical certificate is required from day 3" in combined
    assert "POL-TEST-001" in combined  # metadata table text kept under the title


def test_headings_inside_code_fences_are_ignored() -> None:
    content = (
        "# Real Title\n\n"
        "Intro text.\n\n"
        "```\n"
        "# This is code, not a heading\n"
        "```\n\n"
        "## Real Section\n\nBody.\n"
    )
    parsed = _parse(content)
    headings = [s.heading for s in parsed.sections]
    assert headings == ["Real Title", "Real Section"]
