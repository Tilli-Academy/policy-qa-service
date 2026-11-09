"""Integration tests for the indexing service (ingestion -> embed -> persist -> load).

Uses the deterministic FakeEmbedder over the real policy documents, so no
FastEmbed / network / model download is involved.
"""

from __future__ import annotations

from pathlib import Path

from app.services.indexing import build_and_persist_index, load_index
from tests.conftest import FakeEmbedder


def test_build_persist_and_reload_over_real_documents(real_docs_dir: Path, tmp_path: Path) -> None:
    index_dir = tmp_path / "vector_store"

    store = build_and_persist_index(
        documents_dir=str(real_docs_dir),
        index_dir=str(index_dir),
        embedder=FakeEmbedder(dim=64),
    )
    assert len(store) > 0
    assert (index_dir / "vectors.npy").exists()
    assert (index_dir / "metadata.json").exists()

    reloaded = load_index(index_dir=str(index_dir), embedder=FakeEmbedder(dim=64))
    assert len(reloaded) == len(store)

    # A policy-specific query should surface a chunk from the expected document,
    # with full citation metadata intact.
    results = reloaded.search("annual leave entitlement days", top_k=5)
    assert results
    top = results[0].chunk
    assert top.source_filename.endswith(".md")
    assert top.policy_id is not None
    assert top.section_heading
    assert any(r.chunk.source_filename == "leave_policy.md" for r in results)


def test_reload_search_is_stable_across_restart(real_docs_dir: Path, tmp_path: Path) -> None:
    index_dir = tmp_path / "vector_store"
    build_and_persist_index(
        documents_dir=str(real_docs_dir),
        index_dir=str(index_dir),
        embedder=FakeEmbedder(dim=64),
    )
    # Two independent loads (simulating app restarts) must rank identically.
    a = load_index(index_dir=str(index_dir), embedder=FakeEmbedder(dim=64))
    b = load_index(index_dir=str(index_dir), embedder=FakeEmbedder(dim=64))
    qa = [r.chunk.chunk_index for r in a.search("password mfa security", top_k=5)]
    qb = [r.chunk.chunk_index for r in b.search("password mfa security", top_k=5)]
    assert qa == qb
