"""Offline unit tests for the evaluation package.

No real Groq, no network, no real index: metric math on toy data, dataset loader
validation, a retrieval run over a FakeEmbedder store, and an answer run with a
FakeLLM. The dataset-vs-index consistency test reads the committed metadata.json
(skipped if absent).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.evaluation import metrics
from app.evaluation.answer_eval import run_answer_eval
from app.evaluation.dataset import load_dataset
from app.evaluation.retrieval_eval import render_markdown, run_retrieval_eval
from app.evaluation.schemas import EvalExample
from app.retrieval.retriever import PolicyRetriever
from app.retrieval.vector_store import VectorStore
from app.schemas.ingestion import DocumentChunk
from tests.conftest import FakeEmbedder, FakeLLM, make_search_result

GOLD = {("a.md", "H1"), ("a.md", "H2")}


# -- metric math ----------------------------------------------------------
def test_hit_at_k():
    keys = [("x.md", "Z"), ("a.md", "H1"), ("a.md", "H2")]
    assert metrics.hit_at_k(keys, GOLD, 1) is False
    assert metrics.hit_at_k(keys, GOLD, 2) is True
    assert metrics.hit_at_k(keys, {("none", "none")}, 5) is False


def test_recall_at_k():
    keys = [("a.md", "H1"), ("x.md", "Z"), ("a.md", "H2")]
    assert metrics.recall_at_k(keys, GOLD, 1) == 0.5
    assert metrics.recall_at_k(keys, GOLD, 3) == 1.0
    assert metrics.recall_at_k(keys, set(), 3) == 0.0


def test_first_hit_rank_and_reciprocal_rank():
    keys = [("x.md", "Z"), ("a.md", "H2")]
    assert metrics.first_hit_rank(keys, GOLD) == 2
    assert metrics.reciprocal_rank(keys, GOLD) == 0.5
    assert metrics.first_hit_rank([("x.md", "Z")], GOLD) is None
    assert metrics.reciprocal_rank([("x.md", "Z")], GOLD) == 0.0


def test_keyword_recall_and_citation_and_refusal():
    assert metrics.keyword_recall("You get 25 days", ["25"]) == 1.0
    assert metrics.keyword_recall("MFA is required", ["MFA", "password"]) == 0.5
    assert metrics.keyword_recall("anything", []) == 1.0
    assert metrics.citation_hit([("a.md", "H1")], GOLD) is True
    assert metrics.citation_hit([("z.md", "Q")], GOLD) is False
    assert metrics.refusal_correct(refused=False, answerable=True) is True
    assert metrics.refusal_correct(refused=True, answerable=True) is False
    assert metrics.refusal_correct(refused=True, answerable=False) is True


# -- dataset loader -------------------------------------------------------
def test_default_dataset_loads_and_is_valid():
    examples = load_dataset()
    assert len(examples) >= 10
    ids = [e.id for e in examples]
    assert len(ids) == len(set(ids))
    assert any(e.answerable for e in examples)
    assert any(not e.answerable for e in examples)


def test_schema_rejects_answerable_without_gold():
    with pytest.raises(Exception):
        EvalExample(id="x", question="q", answerable=True, gold_chunks=[])


def test_loader_rejects_duplicate_ids(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(
        '{"id":"a","question":"q","answerable":false}\n'
        '{"id":"a","question":"q2","answerable":false}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_dataset(p)


def test_dataset_gold_headings_exist_in_index():
    meta_path = Path("vector_store/metadata.json")
    if not meta_path.exists():
        pytest.skip("no index metadata available")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    valid = {(c["source_filename"], c["section_heading"]) for c in meta["chunks"]}
    for e in load_dataset():
        for g in e.gold_chunks:
            assert (g.source_filename, g.section_heading) in valid, f"{e.id}: {g}"


# -- end-to-end runs (offline) -------------------------------------------
def _toy_retriever() -> PolicyRetriever:
    chunks = [
        DocumentChunk(source_filename="a.md", policy_id="P", document_title="A",
                      section_heading="Leave", section_hierarchy=["Leave"], chunk_index=0,
                      content="annual leave twenty five days vacation entitlement"),
        DocumentChunk(source_filename="b.md", policy_id="P", document_title="B",
                      section_heading="Security", section_hierarchy=["Security"], chunk_index=0,
                      content="password mfa device security incident"),
    ]
    return PolicyRetriever(VectorStore(FakeEmbedder(dim=64)).build(chunks))


def test_run_retrieval_eval_end_to_end(tmp_path):
    ds = tmp_path / "d.jsonl"
    ds.write_text(
        '{"id":"q1","question":"annual leave days vacation","answerable":true,'
        '"gold_chunks":[{"source_filename":"a.md","section_heading":"Leave"}]}\n'
        '{"id":"ood","question":"irrelevant","answerable":false,"gold_chunks":[]}\n',
        encoding="utf-8",
    )
    report = run_retrieval_eval([1, 3], dataset_path=ds, retriever=_toy_retriever())
    assert report["num_examples"] == 1  # unanswerable item excluded from retrieval eval
    assert report["hit_at_k"][3] == 1.0
    assert 0.0 <= report["mrr"] <= 1.0
    assert isinstance(render_markdown(report), str)


class _DictRetriever:
    """Returns different results per question (for answer-eval refusal paths)."""

    def __init__(self, mapping):
        self._mapping = mapping

    def retrieve(self, question, top_k=None):
        return list(self._mapping.get(question, []))


def test_run_answer_eval_offline(tmp_path):
    ds = tmp_path / "d.jsonl"
    ds.write_text(
        '{"id":"q1","question":"annual leave","answerable":true,'
        '"gold_chunks":[{"source_filename":"leave_policy.md","section_heading":"3. Annual Leave Entitlement"}],'
        '"answer_must_contain":["25"]}\n'
        '{"id":"ood","question":"stocks","answerable":false,"gold_chunks":[]}\n',
        encoding="utf-8",
    )
    retriever = _DictRetriever({
        "annual leave": [make_search_result(
            source_filename="leave_policy.md", section_heading="3. Annual Leave Entitlement", score=0.9)],
        "stocks": [],  # empty -> refusal, no Groq call
    })
    report = run_answer_eval(
        dataset_path=ds, retriever=retriever, llm=FakeLLM("You get 25 days per year."), min_score=0.25
    )
    assert report["num_examples"] == 2
    assert report["groq_calls"] == 1              # only the answerable item hit the LLM
    assert report["refusal_accuracy"] == 1.0      # q1 answered, ood refused
    assert report["citation_accuracy"] == 1.0     # q1 citation matches gold
    assert report["keyword_pass_rate"] == 1.0     # "25" present
