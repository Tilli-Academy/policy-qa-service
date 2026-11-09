"""Tests for the NumPy vector store: normalization, ranking, persistence, errors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.retrieval.embedder import EmbeddingError
from app.retrieval.vector_store import VectorStore
from app.schemas.ingestion import DocumentChunk
from tests.conftest import FakeEmbedder, RaisingEmbedder


def _chunk(content: str, idx: int, heading: str = "Section") -> DocumentChunk:
    return DocumentChunk(
        source_filename="leave_policy.md",
        policy_id="POL-HR-001",
        document_title="Leave Policy",
        section_heading=heading,
        section_hierarchy=["Leave Policy", heading],
        chunk_index=idx,
        content=content,
    )


@pytest.fixture
def built_store() -> VectorStore:
    chunks = [
        _chunk("annual leave entitlement twenty five days", 0, "Annual Leave"),
        _chunk("password requirements mfa device security", 1, "Passwords"),
        _chunk("remote work from home three days per week", 2, "Remote Work"),
    ]
    return VectorStore(FakeEmbedder(dim=64)).build(chunks)


def test_vectors_are_l2_normalized(built_store: VectorStore) -> None:
    norms = np.linalg.norm(built_store._vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_cosine_ranking_puts_most_similar_first(built_store: VectorStore) -> None:
    results = built_store.search("annual leave days", top_k=3)
    assert results[0].chunk.section_heading == "Annual Leave"
    # scores must be monotonically non-increasing
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_top_k_limits_results(built_store: VectorStore) -> None:
    assert len(built_store.search("security password", top_k=1)) == 1
    assert len(built_store.search("security password", top_k=2)) == 2
    # top_k larger than the collection is clamped, not an error.
    assert len(built_store.search("security password", top_k=99)) == 3


def test_metadata_is_preserved_in_results(built_store: VectorStore) -> None:
    result = built_store.search("remote work from home", top_k=1)[0]
    chunk = result.chunk
    assert chunk.source_filename == "leave_policy.md"
    assert chunk.policy_id == "POL-HR-001"
    assert chunk.document_title == "Leave Policy"
    assert chunk.section_heading == "Remote Work"
    assert chunk.section_hierarchy == ["Leave Policy", "Remote Work"]
    assert chunk.chunk_index == 2
    assert isinstance(result.score, float)


def test_persist_and_reload_roundtrip(tmp_path: Path, built_store: VectorStore) -> None:
    out = built_store.save(tmp_path / "index")
    assert (out / "vectors.npy").exists()
    assert (out / "metadata.json").exists()

    reloaded = VectorStore.load(tmp_path / "index", FakeEmbedder(dim=64))
    assert len(reloaded) == len(built_store)
    np.testing.assert_allclose(reloaded._vectors, built_store._vectors, atol=1e-6)
    # Search behaves identically after reload.
    before = built_store.search("annual leave days", top_k=3)
    after = reloaded.search("annual leave days", top_k=3)
    assert [r.chunk.chunk_index for r in before] == [r.chunk.chunk_index for r in after]


def test_empty_store_build_and_search() -> None:
    store = VectorStore(FakeEmbedder(dim=16)).build([])
    assert store.is_empty
    assert len(store) == 0
    assert store.search("anything", top_k=5) == []


def test_empty_store_persist_and_reload(tmp_path: Path) -> None:
    store = VectorStore(FakeEmbedder(dim=16)).build([])
    store.save(tmp_path / "empty")
    reloaded = VectorStore.load(tmp_path / "empty", FakeEmbedder(dim=16))
    assert reloaded.is_empty
    assert reloaded.search("x") == []


def test_load_missing_index_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        VectorStore.load(tmp_path / "nope", FakeEmbedder(dim=16))


def test_load_dimension_mismatch_raises(tmp_path: Path, built_store: VectorStore) -> None:
    built_store.save(tmp_path / "index")  # saved with dim=64
    with pytest.raises(ValueError):
        VectorStore.load(tmp_path / "index", FakeEmbedder(dim=32))


def test_build_dimension_mismatch_raises() -> None:
    # Embedder claims dim=128 but the fake actually emits dim=16 vectors.
    class LyingEmbedder(FakeEmbedder):
        @property
        def dimension(self) -> int:
            return 128

    with pytest.raises(ValueError):
        VectorStore(LyingEmbedder(dim=16)).build([_chunk("text", 0)])


def test_invalid_top_k_raises(built_store: VectorStore) -> None:
    with pytest.raises(ValueError):
        built_store.search("query", top_k=0)
    with pytest.raises(ValueError):
        built_store.search("query", top_k=-3)


def test_embedding_error_is_wrapped_on_build() -> None:
    with pytest.raises(EmbeddingError):
        VectorStore(RaisingEmbedder()).build([_chunk("text", 0)])


def test_embedding_error_is_wrapped_on_search(built_store: VectorStore) -> None:
    # Swap in a failing embedder after a successful build.
    built_store._embedder = RaisingEmbedder()
    with pytest.raises(EmbeddingError):
        built_store.search("query")
