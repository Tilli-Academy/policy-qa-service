"""Abuse-protection helpers for the Q&A API: rate limiting and body-size limits.

Both are FastAPI dependencies attached to the protected router (see
``app/api/routes.py``), and both run AFTER authentication so unauthenticated
requests never consume a rate-limit slot.

Rate limiting is a simple fixed-window counter kept **in memory, per process**.
With multiple uvicorn workers or instances each keeps its own counters, so the
effective global limit is higher than configured — use a shared store (e.g.
Redis) for a true cluster-wide limit.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time

from fastapi import HTTPException, Request, status

from app.api.auth import API_KEY_HEADER_NAME
from app.config import settings

logger = logging.getLogger(__name__)


class RateLimiter:
    """Fixed-window, in-memory rate limiter keyed by an opaque identifier.

    Thread-safe (sync route handlers run in a threadpool). ``now`` is injectable
    so tests can advance time deterministically.
    """

    def __init__(self, max_requests: int, window_seconds: float, now=None) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._now = now or time.monotonic
        self._hits: dict[str, list[float]] = {}  # key -> [window_start, count]
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        """Record a request for ``key``; return False if it exceeds the limit."""
        with self._lock:
            now = self._now()
            entry = self._hits.get(key)
            if entry is None or (now - entry[0]) >= self.window_seconds:
                self._hits[key] = [now, 1]  # start a fresh window
                return True
            if entry[1] >= self.max_requests:
                return False
            entry[1] += 1
            return True

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


# Fallback limiter used when the app lifespan has not populated app.state
# (e.g. in some tests). Lazily created so settings are read at first use.
_default_limiter: RateLimiter | None = None


def _get_default_limiter() -> RateLimiter:
    global _default_limiter
    if _default_limiter is None:
        _default_limiter = RateLimiter(
            settings.rate_limit_requests, settings.rate_limit_window_seconds
        )
    return _default_limiter


def _rate_limit_key(request: Request) -> str:
    """Opaque per-API-key identifier (a short hash, so the key is never stored)."""
    raw = request.headers.get(API_KEY_HEADER_NAME) or ""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def enforce_rate_limit(request: Request) -> None:
    """Dependency: allow or 429 based on the per-API-key fixed-window limit."""
    if not settings.rate_limit_enabled:
        return
    limiter = getattr(request.app.state, "rate_limiter", None) or _get_default_limiter()
    if not limiter.allow(_rate_limit_key(request)):
        logger.warning("Rate limit exceeded for an API key")  # never log the key
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please retry later.",
            headers={"Retry-After": str(settings.rate_limit_window_seconds)},
        )


async def enforce_body_size(request: Request) -> None:
    """Dependency: reject request bodies larger than ``max_request_bytes`` (413).

    Checks the Content-Length header for a fast reject, then the ACTUAL body
    length (so a missing or inaccurate Content-Length cannot bypass the limit).
    """
    max_bytes = settings.max_request_bytes
    content_length = request.headers.get("content-length")
    if content_length is not None and content_length.isdigit() and int(content_length) > max_bytes:
        logger.warning("Request rejected: body too large (content-length)")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Request body too large.",
        )
    body = await request.body()  # cached by Starlette; the route reuses it
    if len(body) > max_bytes:
        logger.warning("Request rejected: body too large")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Request body too large.",
        )
