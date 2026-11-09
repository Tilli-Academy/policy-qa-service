"""Service-level RAG behavior tests: refusal boundaries, citation fidelity, and
sentinel handling. Fully offline (StubRetriever + FakeLLM, no embedder/network).
"""

from __future__ import annotations

from app.llm.prompt import REFUSAL_SENTINEL
from app.services.qa import REFUSAL_MESSAGE, answer_question
from tests.conftest import FakeLLM, StubRetriever, make_search_result


def test_score_exactly_at_floor_is_answered():
    stub = StubRetriever([make_search_result(score=0.25)])
    llm = FakeLLM("an answer")
    resp = answer_question("q", retriever=stub, llm=llm, min_score=0.25)
    assert resp.refused is False          # 0.25 is NOT < 0.25
    assert llm.calls == 1


def test_score_just_below_floor_is_refused_without_llm():
    stub = StubRetriever([make_search_result(score=0.2499)])
    llm = FakeLLM()
    resp = answer_question("q", retriever=stub, llm=llm, min_score=0.25)
    assert resp.refused is True
    assert resp.answer == REFUSAL_MESSAGE
    assert resp.citations == []
    assert llm.calls == 0                 # short-circuited before the LLM


def test_empty_retrieval_refuses_without_llm():
    stub = StubRetriever([])
    llm = FakeLLM()
    resp = answer_question("q", retriever=stub, llm=llm, min_score=0.25)
    assert resp.refused is True
    assert resp.citations == []
    assert llm.calls == 0


def test_citations_count_order_and_fields_match_retrieval():
    results = [
        make_search_result(source_filename="leave_policy.md", section_heading="A",
                           content="c1", score=0.90, chunk_index=0, policy_id="P1"),
        make_search_result(source_filename="it_security_policy.md", section_heading="B",
                           content="c2", score=0.70, chunk_index=3, policy_id="P2"),
        make_search_result(source_filename="leave_policy.md", section_heading="C",
                           content="c3", score=0.55, chunk_index=5, policy_id="P1"),
    ]
    stub = StubRetriever(results)
    resp = answer_question("q", retriever=stub, llm=FakeLLM("ans"), min_score=0.0)

    assert len(resp.citations) == len(results)
    # Order + score fidelity preserved from retrieval (descending).
    assert [c.score for c in resp.citations] == [0.90, 0.70, 0.55]
    for cit, res in zip(resp.citations, results):
        assert cit.source_filename == res.chunk.source_filename
        assert cit.section_heading == res.chunk.section_heading
        assert cit.chunk_index == res.chunk.chunk_index
        assert cit.policy_id == res.chunk.policy_id
        assert cit.score == res.score


def test_sentinel_false_positive_is_not_treated_as_refusal():
    # A legitimate answer that merely MENTIONS the sentinel token mid-text.
    answer = (
        f"The policy does not use the term {REFUSAL_SENTINEL}; "
        "employees receive 25 days of annual leave."
    )
    stub = StubRetriever([make_search_result(score=0.9)])
    resp = answer_question("q", retriever=stub, llm=FakeLLM(answer), min_score=0.0)
    assert resp.refused is False
    assert resp.citations                 # citations retained for a real answer
    assert resp.answer == answer


def test_genuine_sentinel_with_trailing_text_still_refuses():
    stub = StubRetriever([make_search_result(score=0.9)])
    resp = answer_question(
        "q", retriever=stub, llm=FakeLLM(f"{REFUSAL_SENTINEL}."), min_score=0.0
    )
    assert resp.refused is True
    assert resp.citations == []
