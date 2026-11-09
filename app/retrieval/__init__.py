"""Vector embedding and search (Milestone 3).

Public API:
    - ``BaseEmbedder``: injectable embedding interface.
    - ``FastEmbedEmbedder``: FastEmbed (ONNX) backend for all-MiniLM-L6-v2.
    - ``EmbeddingError``: raised on embedding backend failures.
    - ``VectorStore``: NumPy exact cosine search with local persistence.
"""

from __future__ import annotations

from app.retrieval.embedder import BaseEmbedder, EmbeddingError, FastEmbedEmbedder
from app.retrieval.vector_store import VectorStore

__all__ = [
    "BaseEmbedder",
    "FastEmbedEmbedder",
    "EmbeddingError",
    "VectorStore",
]
