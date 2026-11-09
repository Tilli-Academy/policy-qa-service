"""Shared pytest fixtures for the ingestion and retrieval tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from app.llm.base import BaseLLM, LLMError
from app.retrieval.embedder import BaseEmbedder, EmbeddingError
from app.schemas.ingestion import DocumentChunk
from app.schemas.retrieval import SearchResult


class FakeLLM(BaseLLM):
    """Offline LLM stub returning a canned response (no Groq / no network).

    It records the last prompts it received so tests can assert on prompt
    construction and whether it was called at all (refusal short-circuits).
    """

    def __init__(self, response: str = "A canned grounded answer.") -> None:
        self.response = response
        self.calls = 0
        self.last_system_prompt: str | None = None
        self.last_user_prompt: str | None = None

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return self.response


class RaisingLLM(BaseLLM):
    """LLM stub whose calls always fail, to test error handling."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise LLMError("boom")


class FakeEmbedder(BaseEmbedder):
    """Deterministic, offline embedder for tests (no FastEmbed / no network).

    Produces a bag-of-words count vector by hashing each lowercased token into a
    fixed number of buckets. This is fully deterministic across processes (unlike
    Python's built-in ``hash``) and yields higher cosine similarity for texts that
    share more tokens — enough to exercise real ranking behavior.
    """

    def __init__(self, dim: int = 16) -> None:
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.lower().split():
                digest = hashlib.md5(token.encode("utf-8")).hexdigest()
                out[row, int(digest, 16) % self._dim] += 1.0
        return out


class RaisingEmbedder(BaseEmbedder):
    """Embedder whose calls always fail, to test error handling."""

    @property
    def dimension(self) -> int:
        return 8

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        raise RuntimeError("boom")


def make_search_result(
    source_filename: str = "leave_policy.md",
    section_heading: str = "Annual Leave",
    content: str = "Full-time employees receive 25 days of annual leave.",
    score: float = 0.9,
    chunk_index: int = 0,
    policy_id: str | None = "POL-HR-001",
    document_title: str | None = "Leave Policy",
) -> SearchResult:
    """Build a SearchResult with a controllable score + citation metadata.

    Lets behavior tests set exact scores (for refusal-boundary checks) without
    going through an embedder.
    """
    return SearchResult(
        chunk=DocumentChunk(
            source_filename=source_filename,
            policy_id=policy_id,
            document_title=document_title,
            section_heading=section_heading,
            section_hierarchy=[section_heading],
            chunk_index=chunk_index,
            content=content,
        ),
        score=score,
    )


class StubRetriever:
    """Returns a fixed list of SearchResults and records how it was called.

    Duck-typed to match PolicyRetriever.retrieve(); used via dependency overrides
    and direct injection so tests control retrieval without the vector index.
    """

    def __init__(self, results: list[SearchResult]) -> None:
        self._results = list(results)
        self.calls = 0
        self.last_top_k: int | None = None

    def retrieve(self, question: str, top_k: int | None = None) -> list[SearchResult]:
        self.calls += 1
        self.last_top_k = top_k
        return list(self._results)


class RaisingRetriever:
    """Retriever whose retrieve() raises EmbeddingError (to test 502 mapping)."""

    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc or EmbeddingError("embedding backend failed")

    def retrieve(self, question: str, top_k: int | None = None) -> list[SearchResult]:
        raise self._exc


# Project root = parent of the tests/ directory. Used to locate the real sample
# documents without relying on the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_DOCS_DIR = PROJECT_ROOT / "documents" / "policies"


SAMPLE_MD = """# Sample Leave Policy

| Field | Value |
|-------|-------|
| **Policy ID** | POL-TEST-001 |

## 1. Annual Leave

Full-time employees receive 25 days of paid annual leave per year.

### 1.1 Accrual

Leave accrues monthly at 2.08 days per completed month.

## 2. Sick Leave

A medical certificate is required from day 3 of consecutive absence.
"""

NO_HEADING_MD = """This document has no headings at all.
It is just a couple of lines of plain policy text that must be preserved.
"""


@pytest.fixture
def sample_docs_dir(tmp_path: Path) -> Path:
    """A temporary documents directory with a mix of files to ingest/ignore."""
    docs = tmp_path / "policies"
    docs.mkdir()

    # Two valid Markdown files.
    (docs / "sample_policy.md").write_text(SAMPLE_MD, encoding="utf-8")
    (docs / "plain_policy.md").write_text(NO_HEADING_MD, encoding="utf-8")

    # Files that must be ignored:
    (docs / "notes.txt").write_text("not markdown", encoding="utf-8")  # wrong type
    (docs / ".env").write_text("SECRET=value", encoding="utf-8")       # dotfile
    (docs / ".hidden.md").write_text("# Hidden\nhidden", encoding="utf-8")  # hidden md
    (docs / "empty.md").write_text("   \n\n", encoding="utf-8")        # empty/whitespace

    return docs


@pytest.fixture
def real_docs_dir() -> Path:
    """Path to the five committed sample policy documents."""
    if not REAL_DOCS_DIR.is_dir():
        pytest.skip(f"Sample documents directory not found: {REAL_DOCS_DIR}")
    return REAL_DOCS_DIR
