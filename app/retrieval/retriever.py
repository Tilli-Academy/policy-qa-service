"""Retrieval service: a thin, reusable wrapper over the persisted vector index.

This layer deliberately adds no new ranking or embedding logic. It reuses the
Milestone 3 :class:`VectorStore` (loaded via the indexing service) and simply
exposes a question -> ranked chunks operation for the answer pipeline. Keeping it
separate from the vector store lets the LLM layer depend on an interface it can
fake in tests without touching FastEmbed or the index files.
"""

from __future__ import annotations

import logging

from app.config import settings
from app.retrieval.embedder import BaseEmbedder
from app.retrieval.vector_store import VectorStore
from app.schemas.retrieval import SearchResult
from app.services.indexing import load_index

logger = logging.getLogger(__name__)


class PolicyRetriever:
    """Retrieves the most relevant policy chunks for a question.

    Wraps an already-loaded :class:`VectorStore`. Construct directly with a store
    (e.g. in tests), or via :meth:`from_index` to load the persisted index.
    """

    def __init__(self, store: VectorStore) -> None:
        self._store = store

    @classmethod
    def from_index(
        cls,
        index_dir: str | None = None,
        embedder: BaseEmbedder | None = None,
    ) -> "PolicyRetriever":
        """Load the persisted index (read-only) and return a retriever over it."""
        store = load_index(index_dir=index_dir, embedder=embedder)
        return cls(store)

    def retrieve(self, question: str, top_k: int | None = None) -> list[SearchResult]:
        """Return the ``top_k`` chunks most similar to ``question`` (highest score first)."""
        top_k = settings.top_k if top_k is None else top_k
        results = self._store.search(question, top_k=top_k)
        logger.debug("Retrieved %d chunk(s) for question", len(results))
        return results
