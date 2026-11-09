"""Smoke tests confirming Milestone 1 (app + health endpoint) still works.

These avoid an HTTP client dependency by importing the app and calling the
endpoint function directly.
"""

from __future__ import annotations

from app.config import settings
from app.main import app, health


def test_app_metadata_matches_settings() -> None:
    assert app.title == settings.app_name
    assert app.version == settings.app_version


def test_health_returns_consistent_payload() -> None:
    payload = health()
    assert payload == {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
    }
