"""Chunking engine: split sections into size-bounded, overlapping chunks.

Chunking operates one section at a time so a chunk never mixes content from two
different policy sections. Chunks carry the parent document's metadata so each
chunk remains independently citable.
"""

from __future__ import annotations

from app.schemas.ingestion import DocumentChunk, ParsedDocument

# Prefer to cut a chunk at one of these boundaries (strongest first) so chunks
# end at natural points rather than mid-sentence or mid-word.
_BOUNDARIES = ("\n\n", "\n", ". ", " ")


def _split_text(text: str, max_size: int, overlap: int) -> list[str]:
    """Split ``text`` into chunks of at most ``max_size`` characters.

    Consecutive chunks overlap by up to ``overlap`` characters. Where possible a
    chunk ends on a paragraph, line, sentence, or word boundary that falls in the
    second half of the window; otherwise it is cut at ``max_size``. Whitespace-only
    fragments are dropped, so no empty chunks are produced.
    """
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_size:
        return [text]

    # Keep overlap strictly smaller than the window so the cursor always advances.
    overlap = max(0, min(overlap, max_size // 2))

    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_size, length)
        if end < length:
            window = text[start:end]
            cut = -1
            for boundary in _BOUNDARIES:
                found = window.rfind(boundary)
                # Only accept a boundary past the halfway point to avoid tiny chunks.
                if found > max_size // 2:
                    cut = found + len(boundary)
                    break
            if cut != -1:
                end = start + cut

        fragment = text[start:end].strip()
        if fragment:
            chunks.append(fragment)

        if end >= length:
            break
        start = max(end - overlap, start + 1)

    return chunks


def chunk_parsed_document(
    document: ParsedDocument,
    max_chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    """Chunk every section of ``document`` and attach citation metadata.

    ``chunk_index`` is assigned sequentially across the whole document. Sections
    are chunked independently, so content from different sections is never merged
    into the same chunk.
    """
    chunks: list[DocumentChunk] = []
    index = 0
    for section in document.sections:
        for fragment in _split_text(section.content, max_chunk_size, chunk_overlap):
            chunks.append(
                DocumentChunk(
                    source_filename=document.source_filename,
                    policy_id=document.policy_id,
                    document_title=document.document_title,
                    section_heading=section.heading,
                    section_hierarchy=section.hierarchy,
                    chunk_index=index,
                    content=fragment,
                )
            )
            index += 1
    return chunks
