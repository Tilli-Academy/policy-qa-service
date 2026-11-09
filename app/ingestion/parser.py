"""Markdown parser: turn raw Markdown into titled, sectioned content.

Parsing is done with the standard library only (a small line scanner plus
regexes) to avoid adding a Markdown dependency for what is a narrow need:
extracting headings, building their hierarchy, and grouping body text.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.schemas.ingestion import LoadedDocument, ParsedDocument, Section

# A Markdown ATX heading: 1-6 '#', a space, then non-empty text.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(\S.*?)\s*#*\s*$")

# Policy identifiers used across the sample documents, e.g. POL-HR-001.
_POLICY_ID_RE = re.compile(r"\bPOL-[A-Z]+-\d+\b")

# Fenced code-block delimiter (``` or ~~~). Headings inside fences are ignored.
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


def _extract_policy_id(content: str) -> str | None:
    """Return the first policy identifier in the document, if any."""
    match = _POLICY_ID_RE.search(content)
    return match.group(0) if match else None


def parse_document(document: LoadedDocument) -> ParsedDocument:
    """Parse a loaded document into a title, policy id, and ordered sections.

    Sections are delimited by Markdown headings. Each section records its
    immediate heading plus the full ancestor hierarchy (e.g. the document title
    followed by the subsection heading). Content that appears before the first
    heading, and documents with no headings at all, are preserved as a single
    section keyed by the document title (or the filename stem as a fallback), so
    no meaningful policy text is discarded.
    """
    fallback_heading = Path(document.source_filename).stem
    title: str | None = None

    sections: list[Section] = []
    heading_stack: list[tuple[int, str]] = []  # (level, heading) ancestors

    # State for the section currently being accumulated.
    current_level = 0
    current_heading: str | None = None
    current_hierarchy: list[str] = []
    buffer: list[str] = []
    in_fence = False

    def flush() -> None:
        """Emit the buffered lines as a Section if they contain real text."""
        text = "\n".join(buffer).strip()
        buffer.clear()
        if not text:
            return
        heading = current_heading if current_heading is not None else (title or fallback_heading)
        hierarchy = list(current_hierarchy) if current_hierarchy else [heading]
        sections.append(
            Section(level=current_level, heading=heading, hierarchy=hierarchy, content=text)
        )

    for line in document.content.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            buffer.append(line)
            continue

        heading_match = None if in_fence else _HEADING_RE.match(line)
        if heading_match:
            # A new heading closes the previous section.
            flush()
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            if level == 1 and title is None:
                title = heading_text
            # Pop deeper-or-equal headings so the stack holds only ancestors.
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading_text))
            current_level = level
            current_heading = heading_text
            current_hierarchy = [text for _, text in heading_stack]
        else:
            buffer.append(line)

    flush()

    return ParsedDocument(
        source_filename=document.source_filename,
        document_title=title,
        policy_id=_extract_policy_id(document.content),
        sections=sections,
    )
