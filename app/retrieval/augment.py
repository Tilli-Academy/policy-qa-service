"""Heading/title-aware embedding text and boilerplate down-weighting (Milestone 7, P1).

This is a *candidate* retrieval improvement. It is wired only into a separate
index build (``vector_store_v2``) via :func:`build_candidate_index`; the
production index, the default ingestion path, and the running application are
untouched. Two ideas:

  * ``embedding_text`` prepends ``document_title > section_heading`` to the text
    that is embedded, so heading terms contribute to similarity. The stored chunk
    content (used for display/citation) is unchanged.
  * ``boilerplate_penalty`` assigns a small per-chunk score penalty (applied at
    search time) to low-information chunks — each document's title/preamble chunk
    and its "Related Policies" cross-reference section — so they stop crowding
    results for generic or out-of-domain queries.
"""

from __future__ import annotations

import re

from app.schemas.ingestion import DocumentChunk

# Leading section numbering like "8. " or "1.2. " (so "10. Related Policies" and
# "Related Policies" normalize to the same thing).
_NUM_PREFIX = re.compile(r"^\d+(\.\d+)*\.?\s*")

# Score subtracted from a boilerplate chunk's cosine similarity at search time.
BOILERPLATE_PENALTY = 0.15


def embedding_text(chunk: DocumentChunk) -> str:
    """Return the text to embed: ``title > heading`` prefix + the chunk body."""
    title = (chunk.document_title or "").strip()
    heading = (chunk.section_heading or "").strip()
    parts: list[str] = []
    if title:
        parts.append(title)
    if heading and heading != title:  # avoid "Title > Title" for the preamble chunk
        parts.append(heading)
    prefix = " > ".join(parts)
    return f"{prefix}\n\n{chunk.content}" if prefix else chunk.content


def is_boilerplate(chunk: DocumentChunk) -> bool:
    """True for the document title/preamble chunk or a 'Related Policies' section."""
    title = (chunk.document_title or "").strip()
    heading = (chunk.section_heading or "").strip()
    if title and heading == title:
        return True
    normalized = _NUM_PREFIX.sub("", heading).strip().lower()
    return normalized == "related policies"


def boilerplate_penalty(chunk: DocumentChunk) -> float:
    """Per-chunk search-time penalty: ``BOILERPLATE_PENALTY`` for boilerplate, else 0."""
    return BOILERPLATE_PENALTY if is_boilerplate(chunk) else 0.0


def build_candidate_index(
    documents_dir: str | None = None,
    index_dir: str = "vector_store_v2",
    embedder=None,
):
    """Build the candidate (v2) index into ``index_dir`` (default ``vector_store_v2``).

    Uses heading/title-aware embedding text and the boilerplate penalty. Does not
    touch the production index. Requires the FastEmbed ML dependencies (no Groq).
    """
    from app.services.indexing import build_and_persist_index

    return build_and_persist_index(
        documents_dir=documents_dir,
        index_dir=index_dir,
        embedder=embedder,
        embed_text_fn=embedding_text,
        penalty_fn=boilerplate_penalty,
    )


def _main() -> None:  # pragma: no cover - CLI entry point (FastEmbed, no Groq)
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    store = build_candidate_index()
    print(
        f"Built candidate index with {len(store)} chunk(s) at 'vector_store_v2' "
        f"(heading/title-aware embedding; boilerplate penalty={BOILERPLATE_PENALTY})."
    )


if __name__ == "__main__":
    _main()
