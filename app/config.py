"""Centralized application configuration.

Settings are loaded from environment variables (and an optional .env file)
via Pydantic Settings. Keeping configuration here — separate from the FastAPI
application setup in ``app/main.py`` — gives a single, typed source of truth.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings sourced from the environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application metadata (surfaced in the FastAPI OpenAPI docs).
    app_name: str = "Internal Policy Q&A Service"
    app_version: str = "0.1.0"
    app_description: str = (
        "Answers questions using internal Markdown policy documents, "
        "with citations to source filenames and section headings."
    )

    # Location of the Markdown policy documents.
    documents_dir: str = "documents/policies"

    # Ingestion chunking parameters (characters). chunk_size is the maximum
    # size of a chunk; chunk_overlap is how much text is repeated between
    # consecutive chunks of the same section to preserve context at boundaries.
    chunk_size: int = 800
    chunk_overlap: int = 100

    # Embedding / vector search (Milestone 3).
    # embedding_model is the FastEmbed model id; embedding_dim must match the
    # model's output dimension (all-MiniLM-L6-v2 -> 384). index_dir is where the
    # persisted vectors + metadata sidecar are written. top_k is the default
    # number of results returned by a similarity search.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    index_dir: str = "vector_store"
    top_k: int = 5

    # Sub-path the app is mounted under when served behind a reverse proxy
    # (e.g. "/proxy/46453"). Empty for direct/local access. Set via ROOT_PATH.
    # FastAPI uses this to prefix the docs' OpenAPI URL and the "servers" entry,
    # so Swagger UI loads the schema correctly through the proxy.
    root_path: str = ""

    # Groq API key is intentionally OPTIONAL at startup: it is only required
    # once an answer is actually generated. The app must start without it (the
    # LLM client validates the key lazily on first use).
    groq_api_key: str | None = None

    # LLM answer generation (Milestone 4).
    # groq_model is the Groq chat-completions model id. groq_temperature is kept
    # low for faithful, grounded answers. groq_max_tokens caps the response, and
    # groq_timeout_seconds bounds the API call so a hung request cannot block a
    # worker indefinitely.
    groq_model: str = "llama-3.3-70b-versatile"
    groq_temperature: float = 0.1
    groq_max_tokens: int = 1024
    groq_timeout_seconds: float = 30.0

    # Retrieval relevance floor for refusal. If the best chunk's cosine score is
    # below this, the question is treated as unanswerable from the policies and
    # the service refuses WITHOUT calling the LLM. Cosine scores are in [-1, 1].
    answer_min_score: float = 0.25

    # API-key authentication. The expected value for the ``X-API-Key`` header,
    # loaded from API_KEY. When unset the service FAILS CLOSED — protected
    # endpoints reject all requests rather than silently disabling auth. The key
    # is never logged or returned.
    api_key: str | None = None

    # --- Abuse protection -------------------------------------------------
    # Rate limiting for POST /ask, per authenticated API key. The limiter is
    # IN-MEMORY and PER-PROCESS: with multiple uvicorn workers or instances each
    # has its own counters, so the effective global limit is higher. For
    # multi-worker deployments use a shared store (e.g. Redis). 429 on exceed.
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 10
    rate_limit_window_seconds: int = 60

    # Maximum accepted request-body size (bytes) -> 413 when exceeded, and the
    # maximum question length (characters) -> 422 when exceeded.
    max_request_bytes: int = 16384  # 16 KB
    max_question_chars: int = 2000


# Import this singleton elsewhere (e.g. ``from app.config import settings``).
settings = Settings()
