"""Centralized application logging (one setup for the whole service).

A single root handler with a consistent format:

    TIMESTAMP LEVEL logger.name [request_id] message

Request IDs are carried in a :class:`contextvars.ContextVar` and injected into
every log record by :class:`RequestIdFilter`, so all events emitted while handling
one request share the same id (``-`` outside a request). Secrets are never
formatted here — individual call sites are responsible for not passing them.
"""

from __future__ import annotations

import contextvars
import logging
import sys
import uuid

# Per-request id, propagated across the async stack and into sync route handlers
# (anyio copies the context into the threadpool). Defaults to "-" when unset.
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def new_request_id() -> str:
    """Return a short, unique request id."""
    return uuid.uuid4().hex[:12]


class RequestIdFilter(logging.Filter):
    """Attach the current request id to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def configure_logging(level: int = logging.INFO) -> None:
    """Install the root log handler once (idempotent).

    Safe to call from both the app and tests: a guard flag prevents duplicate
    handlers. Noisy third-party HTTP loggers are raised to WARNING so request
    URLs and per-call chatter do not flood the logs.
    """
    root = logging.getLogger()
    if getattr(root, "_app_logging_configured", False):
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    handler.addFilter(RequestIdFilter())
    root.addHandler(handler)
    root.setLevel(level)

    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root._app_logging_configured = True  # type: ignore[attr-defined]
