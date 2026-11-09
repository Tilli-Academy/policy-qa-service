"""Answer service: retrieve policy context, generate a grounded answer, refuse safely.

This is the orchestration layer for Milestone 4. It ties together the reusable
retrieval service (Milestone 3 vector search) and the LLM backend, applying the
refusal policy agreed for this service:

  * Score-floor refusal: if the best retrieved chunk scores below
    ``settings.answer_min_score``, the question is treated as unanswerable from
    the policies and we refuse WITHOUT calling the LLM (cheaper and avoids
    hallucination on clearly-irrelevant questions).
  * Sentinel refusal: otherwise we call the LLM; if it returns the fixed
    ``REFUSAL_SENTINEL`` (context topically close but not actually answering),
    we refuse and drop citations.

Both the retriever and the LLM are injected so the whole flow is testable offline
with fakes (no FastEmbed, no Groq, no network).
"""

from __future__ import annotations

import logging

from app.config import settings
from app.llm.base import BaseLLM
from app.llm.prompt import REFUSAL_SENTINEL, build_messages
from app.retrieval.retriever import PolicyRetriever
from app.schemas.answer import AnswerResponse, Citation

logger = logging.getLogger(__name__)

REFUSAL_MESSAGE = (
    "I could not find enough information in the internal policies to answer that "
    "question."
)


def answer_question(
    question: str,
    *,
    retriever: PolicyRetriever,
    llm: BaseLLM,
    top_k: int | None = None,
    min_score: float | None = None,
) -> AnswerResponse:
    """Answer ``question`` from the policy corpus, or refuse if unsupported.

    Args:
        question: The user's natural-language question.
        retriever: Loaded policy retriever (reuses the persisted vector index).
        llm: The LLM backend used to synthesize the answer.
        top_k: Number of chunks to retrieve; defaults to the configured TOP_K.
        min_score: Relevance floor for refusal; defaults to ANSWER_MIN_SCORE.

    Returns:
        An :class:`AnswerResponse`. ``refused`` is True (with no citations and the
        standard refusal message) when the policies lack the information.

    Raises:
        EmbeddingError: if query embedding fails (propagated from the retriever).
        LLMError: if the LLM call fails (propagated from the backend).
    """
    min_score = settings.answer_min_score if min_score is None else min_score

    results = retriever.retrieve(question, top_k=top_k)
    logger.info(
        "Retrieval complete: chunks=%d top_score=%s",
        len(results),
        f"{results[0].score:.4f}" if results else "n/a",
    )

    # Score-floor refusal: nothing retrieved, or the best hit is too weak.
    best_score = results[0].score if results else float("-inf")
    if not results or best_score < min_score:
        logger.info("Refusing (low relevance): best_score=%.4f < min_score=%.4f", best_score, min_score)
        return AnswerResponse(
            question=question,
            answer=REFUSAL_MESSAGE,
            refused=True,
            citations=[],
        )

    system_prompt, user_prompt = build_messages(question, results)
    raw = llm.generate(system_prompt, user_prompt)

    # Sentinel refusal: the model judged the context insufficient. The system
    # prompt instructs the model to reply with ONLY the sentinel token, so we
    # match it as a leading token (after stripping) rather than via substring
    # containment — otherwise a valid answer that merely *mentions* the token
    # would be misclassified as a refusal (false positive).
    if raw.strip().startswith(REFUSAL_SENTINEL):
        logger.info("Refusing (model sentinel) for question")
        return AnswerResponse(
            question=question,
            answer=REFUSAL_MESSAGE,
            refused=True,
            citations=[],
        )

    citations = [Citation.from_search_result(r) for r in results]
    return AnswerResponse(
        question=question,
        answer=raw,
        refused=False,
        citations=citations,
    )
