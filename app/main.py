"""FastAPI application entry point.

Wires together the service: a health probe (Milestone 1) plus the question
-answering API (Milestone 4). The vector index and LLM backend are constructed
once at startup via the lifespan and stored on ``app.state`` for the routes to
reuse. Startup is fail-soft: if the index or Groq backend cannot be initialized,
the app still starts (so ``/health`` works) and ``/ask`` returns a 503 until the
dependency is fixed.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.api.routes import router as qa_router
from app.config import settings
from app.logging_config import configure_logging, new_request_id, request_id_var

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the retriever and LLM backend once, tolerating failures."""
    logger.info("Application startup: initializing retriever and LLM backend")
    # Retriever: load the persisted vector index (read-only). Heavy import kept
    # local so importing this module never pulls in FastEmbed.
    app.state.retriever = None
    app.state.llm = None
    try:
        from app.retrieval.retriever import PolicyRetriever

        app.state.retriever = PolicyRetriever.from_index()
        logger.info("Vector index loaded")
    except Exception as exc:  # pragma: no cover - startup-path, environment dependent
        logger.error("Could not load the vector index at startup: %s", type(exc).__name__)

    try:
        from app.llm.groq_client import GroqLLM

        app.state.llm = GroqLLM()
        logger.info("LLM backend initialized (model=%s)", app.state.llm.model)
    except Exception as exc:  # pragma: no cover - startup-path, environment dependent
        # GroqLLM raises a secret-safe LLMError when the key is missing.
        logger.error("Could not initialize the LLM backend at startup: %s", type(exc).__name__)

    # In-memory, per-process rate limiter shared across requests.
    from app.api.limits import RateLimiter

    app.state.rate_limiter = RateLimiter(
        settings.rate_limit_requests, settings.rate_limit_window_seconds
    )
    logger.info(
        "Rate limiter ready (%d requests / %ds per API key, in-memory per process)",
        settings.rate_limit_requests,
        settings.rate_limit_window_seconds,
    )

    logger.info("Application startup complete")
    yield
    logger.info("Application shutdown")


app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    # When behind a reverse proxy under a sub-path, root_path makes FastAPI
    # prefix the OpenAPI URL in the docs and the "servers" entry. Empty by
    # default, so local/direct access is unaffected.
    root_path=settings.root_path,
    lifespan=lifespan,
)

app.include_router(qa_router)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Assign a request id, log request start/complete + duration, trace errors.

    Only the method and path are logged — never the request body (which may
    contain the employee's question). The id is returned in ``X-Request-ID`` and
    is reused from an incoming ``X-Request-ID`` header when present.
    """
    rid = request.headers.get("X-Request-ID") or new_request_id()
    token = request_id_var.set(rid)
    start = time.perf_counter()
    logger.info("request start method=%s path=%s", request.method, request.url.path)
    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "request complete path=%s status=%d duration_ms=%.1f",
            request.url.path,
            response.status_code,
            duration_ms,
        )
        response.headers["X-Request-ID"] = rid
        return response
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "Unhandled error path=%s duration_ms=%.1f", request.url.path, duration_ms
        )
        raise
    finally:
        request_id_var.reset(token)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Liveness probe returning a consistent JSON payload."""
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
    }
