"""HTTP routes for question answering (Milestone 4).

The retriever and LLM backend are built once at application startup (see the
lifespan in ``app.main``) and stashed on ``app.state``. The dependency providers
below surface them to the ``/ask`` handler and translate missing/failed
infrastructure into clean HTTP status codes. Nothing here logs or returns the
Groq API key or raw backend internals.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.auth import require_api_key
from app.api.limits import enforce_body_size, enforce_rate_limit
from app.llm.base import BaseLLM, LLMError, LLMTimeoutError
from app.retrieval.embedder import EmbeddingError
from app.retrieval.retriever import PolicyRetriever
from app.schemas.answer import AnswerResponse, AskRequest
from app.services.qa import answer_question

logger = logging.getLogger(__name__)

# All question-answering routes are protected. Dependencies run in order:
#   1. require_api_key   -> authenticate first (401)
#   2. enforce_body_size -> reject oversized bodies (413)
#   3. enforce_rate_limit-> per-API-key rate limit, only after auth passes (429)
# /health is registered on the app directly (app/main.py) and stays public.
router = APIRouter(
    tags=["qa"],
    dependencies=[
        Depends(require_api_key),
        Depends(enforce_body_size),
        Depends(enforce_rate_limit),
    ],
)


def get_retriever(request: Request) -> PolicyRetriever:
    """Return the app's retriever, or 503 if the index failed to load at startup."""
    retriever = getattr(request.app.state, "retriever", None)
    if retriever is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval index is not available.",
        )
    return retriever


def get_llm(request: Request) -> BaseLLM:
    """Return the app's LLM backend, or 503 if it was not configured at startup."""
    llm = getattr(request.app.state, "llm", None)
    if llm is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Answer generation backend is not available.",
        )
    return llm


@router.post("/ask", response_model=AnswerResponse)
def ask(
    payload: AskRequest,
    retriever: PolicyRetriever = Depends(get_retriever),
    llm: BaseLLM = Depends(get_llm),
) -> AnswerResponse:
    """Answer a policy question from the indexed documents, with citations."""
    try:
        return answer_question(
            payload.question,
            retriever=retriever,
            llm=llm,
            top_k=payload.top_k,
        )
    except EmbeddingError:
        logger.error("Embedding failure while answering a question")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to process the question (embedding backend error).",
        )
    except LLMTimeoutError:
        # Already logged (type + duration) in the Groq client. Map to 504 so the
        # client knows it timed out; never expose internal details or the key.
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The answer service timed out. Please try again.",
        )
    except LLMError:
        # Already logged with type + duration in the Groq client; avoid a
        # duplicate log here and just map to a generic, secret-safe 502.
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to generate an answer (LLM backend error).",
        )
