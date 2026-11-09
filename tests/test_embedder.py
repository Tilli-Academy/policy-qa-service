"""Tests for the embedding interface (using the deterministic fake backend)."""

from __future__ import annotations

import numpy as np

from app.retrieval.embedder import BaseEmbedder
from tests.conftest import FakeEmbedder


def test_fake_embedder_is_a_base_embedder() -> None:
    assert isinstance(FakeEmbedder(), BaseEmbedder)


def test_embed_documents_shape_matches_dimension() -> None:
    emb = FakeEmbedder(dim=16)
    vectors = emb.embed_documents(["hello world", "another text", "third one"])
    assert vectors.shape == (3, 16)
    assert vectors.dtype == np.float32


def test_embed_empty_batch_returns_empty_matrix() -> None:
    emb = FakeEmbedder(dim=16)
    vectors = emb.embed_documents([])
    assert vectors.shape == (0, 16)


def test_embedding_is_deterministic() -> None:
    emb = FakeEmbedder(dim=16)
    a = emb.embed_documents(["annual leave entitlement"])
    b = emb.embed_documents(["annual leave entitlement"])
    assert np.array_equal(a, b)


def test_embed_query_returns_single_vector() -> None:
    emb = FakeEmbedder(dim=16)
    vector = emb.embed_query("password requirements")
    assert vector.shape == (16,)


def test_similar_texts_have_higher_dot_product_than_dissimilar() -> None:
    emb = FakeEmbedder(dim=64)
    q = emb.embed_query("annual leave days")
    similar = emb.embed_query("annual leave entitlement days")
    different = emb.embed_query("password mfa security incident")
    assert float(q @ similar) > float(q @ different)
