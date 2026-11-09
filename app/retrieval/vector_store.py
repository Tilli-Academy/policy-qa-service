"""In-memory vector store with NumPy exact cosine search and local persistence.

Vectors are L2-normalized at build/load time, so cosine similarity reduces to a
single matrix-vector dot product. The store keeps the original
:class:`DocumentChunk` for every vector, so search results carry full citation
metadata. Persistence is a ``.npy`` vector file plus a JSON metadata sidecar.

FAISS is intentionally not used: at the current corpus size exact NumPy search is
both simpler and equivalent. The public interface (build/search/save/load) is
designed so a FAISS-backed implementation can replace it later without changing
callers.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from app.config import settings
from app.retrieval.embedder import BaseEmbedder, EmbeddingError
from app.schemas.ingestion import DocumentChunk
from app.schemas.retrieval import SearchResult

_VECTORS_FILE = "vectors.npy"
_METADATA_FILE = "metadata.json"


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """Return ``matrix`` with each row scaled to unit L2 norm (zero rows left as-is)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32)


class VectorStore:
    """Holds normalized chunk vectors and supports cosine similarity search."""

    def __init__(self, embedder: BaseEmbedder) -> None:
        self._embedder = embedder
        self._dimension = embedder.dimension
        self._vectors: np.ndarray = np.empty((0, self._dimension), dtype=np.float32)
        self._chunks: list[DocumentChunk] = []
        # Per-chunk search-time score penalties (zeros => no effect). Populated by
        # build()/load(); the production index has none, so behavior is unchanged.
        self._penalties: np.ndarray = np.zeros(0, dtype=np.float32)

    # -- introspection -----------------------------------------------------
    @property
    def dimension(self) -> int:
        return self._dimension

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def is_empty(self) -> bool:
        return len(self._chunks) == 0

    # -- build -------------------------------------------------------------
    def build(
        self,
        chunks: list[DocumentChunk],
        embed_text_fn: Callable[[DocumentChunk], str] | None = None,
        penalty_fn: Callable[[DocumentChunk], float] | None = None,
    ) -> "VectorStore":
        """Embed and store ``chunks``. An empty list yields an empty (valid) store.

        ``embed_text_fn`` optionally maps a chunk to the *text that is embedded*
        (e.g. heading/title-augmented) while the original chunk content is still
        stored for display/citation. ``penalty_fn`` optionally assigns a per-chunk
        score penalty subtracted at search time (to down-weight boilerplate). Both
        default to identity/zero behavior, so existing callers are unaffected.
        """
        self._chunks = list(chunks)
        self._penalties = np.array(
            [float(penalty_fn(c)) if penalty_fn else 0.0 for c in self._chunks],
            dtype=np.float32,
        )
        texts = [embed_text_fn(c) if embed_text_fn else c.content for c in self._chunks]
        if not texts:
            self._vectors = np.empty((0, self._dimension), dtype=np.float32)
            return self

        try:
            vectors = self._embedder.embed_documents(texts)
        except EmbeddingError:
            raise
        except Exception as exc:  # defensive: wrap any backend failure
            raise EmbeddingError(f"Embedding failed during build: {exc}") from exc

        vectors = np.asarray(vectors, dtype=np.float32)
        self._validate_matrix(vectors, expected_rows=len(texts))
        self._vectors = _normalize(vectors)
        return self

    # -- search ------------------------------------------------------------
    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Return the ``top_k`` chunks most cosine-similar to ``query``."""
        top_k = settings.top_k if top_k is None else top_k
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if self.is_empty:
            return []

        try:
            raw = self._embedder.embed_query(query)
        except EmbeddingError:
            raise
        except Exception as exc:  # defensive: wrap any backend failure
            raise EmbeddingError(f"Embedding failed during query: {exc}") from exc

        vector = np.asarray(raw, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self._dimension:
            raise ValueError(
                f"Query embedding dimension {vector.shape[0]} does not match "
                f"store dimension {self._dimension}"
            )
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector = vector / norm

        scores = self._vectors @ vector  # cosine similarity (both sides normalized)
        scores = scores - self._penalties  # down-weight penalized chunks (zeros by default)
        k = min(top_k, scores.shape[0])
        # argsort descending, stable enough for deterministic ties at this scale.
        order = np.argsort(-scores, kind="stable")[:k]
        return [SearchResult(chunk=self._chunks[i], score=float(scores[i])) for i in order]

    # -- persistence -------------------------------------------------------
    def save(self, index_dir: str | Path | None = None) -> Path:
        """Persist vectors (``.npy``) and metadata (JSON) to ``index_dir``."""
        path = Path(index_dir if index_dir is not None else settings.index_dir)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / _VECTORS_FILE, self._vectors)
        metadata = {
            "embedding_model": getattr(self._embedder, "model_name", None),
            "dimension": self._dimension,
            "count": len(self._chunks),
            "penalties": self._penalties.tolist(),
            "chunks": [chunk.model_dump() for chunk in self._chunks],
        }
        (path / _METADATA_FILE).write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, index_dir: str | Path, embedder: BaseEmbedder) -> "VectorStore":
        """Load a persisted store, validating counts and dimensions.

        Raises:
            FileNotFoundError: if the vector or metadata file is missing.
            ValueError: if the stored data is inconsistent or dimension mismatches
                the provided embedder.
        """
        path = Path(index_dir)
        vectors_path = path / _VECTORS_FILE
        metadata_path = path / _METADATA_FILE
        if not vectors_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"No vector index found in {path} (expected {_VECTORS_FILE} and {_METADATA_FILE})."
            )

        store = cls(embedder)
        vectors = np.load(vectors_path).astype(np.float32)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        chunks = [DocumentChunk(**item) for item in metadata.get("chunks", [])]

        if vectors.ndim != 2:
            raise ValueError(f"Stored vectors must be 2-D, got shape {vectors.shape}")
        if vectors.shape[0] != len(chunks):
            raise ValueError(
                f"Vector/metadata count mismatch: {vectors.shape[0]} vectors vs {len(chunks)} chunks"
            )
        if vectors.shape[0] > 0 and vectors.shape[1] != embedder.dimension:
            raise ValueError(
                f"Stored dimension {vectors.shape[1]} does not match embedder dimension "
                f"{embedder.dimension}"
            )

        penalties_raw = metadata.get("penalties")
        if penalties_raw is None:
            # Older/production indexes have no penalties -> zeros (no effect).
            penalties = np.zeros(len(chunks), dtype=np.float32)
        else:
            penalties = np.asarray(penalties_raw, dtype=np.float32)
            if penalties.shape[0] != len(chunks):
                raise ValueError(
                    f"Penalty/metadata count mismatch: {penalties.shape[0]} penalties "
                    f"vs {len(chunks)} chunks"
                )

        store._vectors = vectors if vectors.shape[0] > 0 else np.empty(
            (0, embedder.dimension), dtype=np.float32
        )
        store._chunks = chunks
        store._penalties = penalties
        store._dimension = embedder.dimension
        return store

    # -- internal ----------------------------------------------------------
    def _validate_matrix(self, vectors: np.ndarray, expected_rows: int) -> None:
        if vectors.ndim != 2 or vectors.shape[0] != expected_rows:
            raise ValueError(
                f"Embedder returned shape {vectors.shape}; expected ({expected_rows}, "
                f"{self._dimension})"
            )
        if vectors.shape[1] != self._dimension:
            raise ValueError(
                f"Embedding dimension {vectors.shape[1]} does not match configured "
                f"dimension {self._dimension}"
            )
