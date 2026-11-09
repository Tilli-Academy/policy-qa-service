"""API-key authentication for the policy Q&A endpoints.

Uses FastAPI's :class:`APIKeyHeader` to require an ``X-API-Key`` request header.
The expected key is read from settings (``API_KEY``, sourced from the
environment / .env) — never hardcoded. Validation is constant-time.

Security properties:
  * **Fail closed:** if no key is configured server-side, protected requests are
    rejected (503) rather than silently allowed through.
  * **401 for missing/invalid** client keys.
  * The key value is never logged, returned, or placed in error details.

``auto_error=False`` lets us return a 401 for a *missing* header (FastAPI's
default would be 403).
"""

from __future__ import annotations

import hmac
import logging

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.config import settings

logger = logging.getLogger(__name__)

API_KEY_HEADER_NAME = "X-API-Key"

api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


def require_api_key(provided_key: str | None = Security(api_key_header)) -> None:
    """Dependency enforcing a valid ``X-API-Key`` header (constant-time compare)."""
    expected = settings.api_key
    if not expected:
        # Authentication is required but not configured: fail closed. Logs the
        # condition, never any key material.
        logger.error("Authentication not configured; rejecting request (fail closed)")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured.",
        )
    if provided_key is None or not hmac.compare_digest(provided_key, expected):
        # Log the failure only — never the provided or expected key.
        logger.warning("Authentication failed: missing or invalid X-API-Key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": API_KEY_HEADER_NAME},
        )
