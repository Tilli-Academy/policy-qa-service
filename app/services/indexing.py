"""Indexing service: build/persist and load the policy vector index.

Ties the Milestone 2 ingestion pipeline to the Milestone 3 vector store. The
embedder is injectable; it defaults to :class:`FastEmbedEmbedder`, which requires
the ML dependencies (``requirements-ml.txt``). Tests inject a fake embedder, so
they never touch FastEmbed.
"""

from __future__ import annotations

import logging

from app.config import settings
from app.ingestion import ingest_documents
from app.retrieval.embedder import BaseEmbedder
from app.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _default_embedder() -> BaseEmbedder:
    # Imported here (not at call sites) so the heavy backend is only constructed
    # when an embedder is not supplied explicitly.
    from app.retrieval.embedder import FastEmbedEmbedder

    return FastEmbedEmbedder()


def build_and_persist_index(
    documents_dir: str | None = None,
    index_dir: str | None = None,
    embedder: BaseEmbedder | None = None,
    embed_text_fn=None,
    penalty_fn=None,
) -> VectorStore:
    """Ingest documents, embed their chunks, and persist the vector index.

    ``embed_text_fn``/``penalty_fn`` are optional hooks (default None => current
    behavior) used by the candidate (v2) index build; see ``app.retrieval.augment``.
    """
    embedder = embedder or _default_embedder()
    index_dir = index_dir if index_dir is not None else settings.index_dir

    chunks = ingest_documents(documents_dir)
    store = VectorStore(embedder).build(chunks, embed_text_fn=embed_text_fn, penalty_fn=penalty_fn)
    path = store.save(index_dir)
    logger.info("Built index of %d chunk(s) and saved to %s", len(store), path)
    return store


def load_index(
    index_dir: str | None = None,
    embedder: BaseEmbedder | None = None,
) -> VectorStore:
    """Load a previously persisted vector index."""
    embedder = embedder or _default_embedder()
    index_dir = index_dir if index_dir is not None else settings.index_dir
    store = VectorStore.load(index_dir, embedder)
    logger.info("Loaded index of %d chunk(s) from %s", len(store), index_dir)
    return store


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build or query the policy vector index.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="Ingest documents and persist the vector index.")
    search_parser = sub.add_parser("search", help="Load the index and run a query.")
    search_parser.add_argument("query", help="Query text.")
    search_parser.add_argument("-k", "--top-k", type=int, default=settings.top_k)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "build":
        store = build_and_persist_index()
        print(f"Built and saved index with {len(store)} chunk(s) to '{settings.index_dir}'.")
    elif args.command == "search":
        store = load_index()
        for rank, result in enumerate(store.search(args.query, top_k=args.top_k), start=1):
            c = result.chunk
            print(f"\n#{rank}  score={result.score:.4f}  [{c.source_filename} :: {c.section_heading}]")
            print(f"    policy_id={c.policy_id}  chunk_index={c.chunk_index}")
            print(f"    {c.content[:200]}{'...' if len(c.content) > 200 else ''}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    _main()
