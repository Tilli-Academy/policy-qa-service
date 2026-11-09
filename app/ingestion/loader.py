"""Document loader: discover and read Markdown policy files from disk.

Responsibilities are limited to the filesystem: finding the right files and
reading their bytes as UTF-8. Parsing and chunking live in sibling modules.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.schemas.ingestion import LoadedDocument

logger = logging.getLogger(__name__)

# Only Markdown files are ingested. Anything else (e.g. .env, .txt, images)
# is ignored by construction because discovery globs for this suffix only.
MARKDOWN_SUFFIX = ".md"


def discover_markdown_files(documents_dir: str | Path) -> list[Path]:
    """Return a sorted list of Markdown files under ``documents_dir``.

    Hidden files and files inside hidden directories (names starting with ``.``)
    are skipped, so dotfiles such as ``.env`` are never returned. Non-Markdown
    files are excluded because only ``*.md`` is matched.

    Raises:
        FileNotFoundError: if ``documents_dir`` does not exist.
        NotADirectoryError: if ``documents_dir`` exists but is not a directory.
    """
    base = Path(documents_dir)
    if not base.exists():
        raise FileNotFoundError(f"Documents directory not found: {base}")
    if not base.is_dir():
        raise NotADirectoryError(f"Documents path is not a directory: {base}")

    files: list[Path] = []
    for path in sorted(base.rglob(f"*{MARKDOWN_SUFFIX}")):
        if not path.is_file():
            continue
        # Skip hidden files and any file nested under a hidden directory.
        relative_parts = path.relative_to(base).parts
        if any(part.startswith(".") for part in relative_parts):
            continue
        files.append(path)
    return files


def load_documents(documents_dir: str | Path) -> list[LoadedDocument]:
    """Load all Markdown documents under ``documents_dir`` as UTF-8 text.

    Empty (whitespace-only) and unreadable files are skipped with a warning
    rather than raising, so a single bad file does not abort ingestion.
    A missing directory still raises (via :func:`discover_markdown_files`),
    because that indicates a configuration problem.
    """
    documents: list[LoadedDocument] = []
    for path in discover_markdown_files(documents_dir):
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("Skipping unreadable document %s: %s", path, exc)
            continue
        if not content.strip():
            logger.warning("Skipping empty document: %s", path)
            continue
        documents.append(LoadedDocument(source_filename=path.name, content=content))
    return documents
