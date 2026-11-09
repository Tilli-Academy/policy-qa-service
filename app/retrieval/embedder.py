"""Embedding interface and the FastEmbed-backed implementation.

The :class:`BaseEmbedder` abstraction lets the vector store depend on an
*interface* rather than a concrete backend, so the FastEmbed backend can be
swapped (e.g. for a fake in tests, or a different model later) without touching
callers. The heavy ``fastembed`` dependency is imported lazily inside
:class:`FastEmbedEmbedder` so merely importing this module never requires it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from app.config import settings


class EmbeddingError(RuntimeError):
    """Raised when an embedding backend fails to produce embeddings."""


class BaseEmbedder(ABC):
    """Abstract embedding backend producing float32 vectors.

    Implementations must return L2-comparable vectors (normalization is handled
    by the vector store, so backends need not normalize themselves).
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The dimensionality of the vectors this backend produces."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts, returning an ``(len(texts), dimension)`` float32 array."""

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string, returning a ``(dimension,)`` float32 array.

        Defaults to reusing :meth:`embed_documents`; backends may override if they
        offer a dedicated query encoding.
        """
        return self.embed_documents([text])[0]


class FastEmbedEmbedder(BaseEmbedder):
    """Embedding backend using FastEmbed (ONNX runtime).

    Wraps ``fastembed.TextEmbedding`` for the configured model
    (``sentence-transformers/all-MiniLM-L6-v2`` -> 384 dims). ``fastembed`` is
    imported lazily so this class only requires it when actually instantiated.
    """

    def __init__(
        self,
        model_name: str | None = None,
        dimension: int | None = None,
        cache_dir: str | None = None,
    ) -> None:
        self.model_name = model_name or settings.embedding_model
        self._dimension = dimension or settings.embedding_dim
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - exercised only without fastembed
            raise ImportError(
                "fastembed is not installed. Install the ML dependencies with "
                "`pip install -r requirements-ml.txt` to use FastEmbedEmbedder."
            ) from exc
        self._model = TextEmbedding(model_name=self.model_name, cache_dir=cache_dir)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)
        try:
            vectors = list(self._model.embed(texts))
        except Exception as exc:  # pragma: no cover - backend/runtime failures
            raise EmbeddingError(f"FastEmbed failed to embed {len(texts)} text(s): {exc}") from exc
        return np.asarray(np.vstack(vectors), dtype=np.float32)
