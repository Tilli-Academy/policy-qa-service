"""Offline tests for the Milestone 7 P1 candidate-index behavior.

Covers heading/title-aware embedding text, boilerplate detection/penalty, and the
backward-compatible VectorStore build/search/persist hooks. Uses FakeEmbedder —
no FastEmbed, no network, no Groq.
"""

from __future__ import annotations

import pytest

from app.retrieval import augment
from app.retrieval.vector_store import VectorStore
from app.schemas.ingestion import DocumentChunk
from tests.conftest import FakeEmbedder


def _chunk(title, heading, content, idx=0) -> DocumentChunk:
    return DocumentChunk(
        source_filename="f.md", policy_id="P", document_title=title,
        section_heading=heading, section_hierarchy=[heading], chunk_index=idx, content=content,
    )


# -- embedding_text -------------------------------------------------------
def test_embedding_text_prepends_title_and_heading():
    t = augment.embedding_text(_chunk("Leave Policy", "3. Annual Leave", "Full-time staff get 25 days."))
    assert t.startswith("Leave Policy > 3. Annual Leave\n\n")
    assert "25 days" in t


def test_embedding_text_dedupes_title_preamble_chunk():
    t = augment.embedding_text(_chunk("Leave Policy", "Leave Policy", "preamble table"))
    first_line = t.split("\n\n")[0]
    assert first_line == "Leave Policy"  # not "Leave Policy > Leave Policy"


def test_embedding_text_without_title():
    t = augment.embedding_text(_chunk(None, "Heading", "body"))
    assert t.startswith("Heading\n\n")


# -- boilerplate detection + penalty -------------------------------------
def test_is_boilerplate():
    assert augment.is_boilerplate(_chunk("Leave Policy", "Leave Policy", "x")) is True
    assert augment.is_boilerplate(_chunk("Leave Policy", "10. Related Policies", "x")) is True
    assert augment.is_boilerplate(_chunk("Leave Policy", "Related Policies", "x")) is True
    assert augment.is_boilerplate(_chunk("Leave Policy", "3. Annual Leave", "x")) is False


def test_boilerplate_penalty_values():
    assert augment.boilerplate_penalty(_chunk("P", "10. Related Policies", "x")) == augment.BOILERPLATE_PENALTY
    assert augment.boilerplate_penalty(_chunk("P", "2. Definitions", "x")) == 0.0


# -- VectorStore hooks (backward compatible) ------------------------------
def test_build_with_embed_text_fn_uses_heading_terms():
    # Body lacks the word "approval"; the heading supplies it. Heading-aware
    # embedding should let a query for "approval process" match this chunk.
    chunks = [
        _chunk("Remote Policy", "7. Remote Work Approval Process", "submit request in peoplehub", 0),
        _chunk("Remote Policy", "2. Definitions", "hybrid working mix office remote", 1),
    ]
    store = VectorStore(FakeEmbedder(dim=64)).build(chunks, embed_text_fn=augment.embedding_text)
    res = store.search("approval process", top_k=2)
    assert res[0].chunk.section_heading == "7. Remote Work Approval Process"


def test_penalty_subtracts_from_score_and_persists(tmp_path):
    chunks = [
        _chunk("P", "Related Policies", "see other policies cross reference", 0),
        _chunk("P", "3. Content", "unique token zebra", 1),
    ]
    query = "other policies cross reference"

    baseline = VectorStore(FakeEmbedder(dim=64)).build(chunks)  # no penalty
    raw = {r.chunk.section_heading: r.score for r in baseline.search(query, top_k=2)}

    penalized = VectorStore(FakeEmbedder(dim=64)).build(chunks, penalty_fn=augment.boilerplate_penalty)
    got = {r.chunk.section_heading: r.score for r in penalized.search(query, top_k=2)}

    # Boilerplate score reduced by exactly BOILERPLATE_PENALTY; non-boilerplate unchanged.
    assert got["Related Policies"] == pytest.approx(raw["Related Policies"] - augment.BOILERPLATE_PENALTY, abs=1e-5)
    assert got["3. Content"] == pytest.approx(raw["3. Content"], abs=1e-5)

    # Penalties survive a save/load round-trip.
    penalized.save(tmp_path)
    reloaded = VectorStore.load(tmp_path, FakeEmbedder(dim=64))
    got2 = {r.chunk.section_heading: r.score for r in reloaded.search(query, top_k=2)}
    assert got2["Related Policies"] == pytest.approx(got["Related Policies"], abs=1e-5)


def test_default_build_has_no_penalty_effect(tmp_path):
    # Regression guard: default build (no hooks) behaves exactly as before and
    # loads back with zero penalties.
    chunks = [_chunk("P", "A", "alpha beta", 0), _chunk("P", "B", "gamma delta", 1)]
    store = VectorStore(FakeEmbedder(dim=64)).build(chunks)
    store.save(tmp_path)
    reloaded = VectorStore.load(tmp_path, FakeEmbedder(dim=64))
    a = [(r.chunk.section_heading, round(r.score, 6)) for r in store.search("alpha beta", top_k=2)]
    b = [(r.chunk.section_heading, round(r.score, 6)) for r in reloaded.search("alpha beta", top_k=2)]
    assert a == b
