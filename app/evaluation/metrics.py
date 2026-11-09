"""Pure metric functions for RAG evaluation.

All functions operate on plain data (lists of ``(source_filename,
section_heading)`` keys and sets of gold keys) so they are deterministic and
unit-testable without a retriever, index, or LLM.
"""

from __future__ import annotations

from collections.abc import Iterable

Key = tuple[str, str]


def _mean(values: Iterable[float]) -> float:
    items = [float(v) for v in values]
    return sum(items) / len(items) if items else 0.0


# -- retrieval metrics ----------------------------------------------------
def hit_at_k(retrieved_keys: list[Key], gold: set[Key], k: int) -> bool:
    """True if any of the top-``k`` retrieved keys is a gold key."""
    return any(key in gold for key in retrieved_keys[:k])


def recall_at_k(retrieved_keys: list[Key], gold: set[Key], k: int) -> float:
    """Fraction of distinct gold keys present in the top-``k`` retrieved keys."""
    if not gold:
        return 0.0
    found = set(retrieved_keys[:k]) & gold
    return len(found) / len(gold)


def first_hit_rank(retrieved_keys: list[Key], gold: set[Key]) -> int | None:
    """1-based rank of the first gold key in ``retrieved_keys`` (None if absent)."""
    for rank, key in enumerate(retrieved_keys, start=1):
        if key in gold:
            return rank
    return None


def reciprocal_rank(retrieved_keys: list[Key], gold: set[Key]) -> float:
    """Reciprocal of the first-hit rank; 0.0 if no gold key is retrieved."""
    rank = first_hit_rank(retrieved_keys, gold)
    return 1.0 / rank if rank else 0.0


# -- answer metrics -------------------------------------------------------
def citation_hit(citation_keys: list[Key], gold: set[Key]) -> bool:
    """True if any returned citation key matches a gold key."""
    return any(key in gold for key in citation_keys)


def keyword_recall(answer: str, keywords: list[str]) -> float:
    """Fraction of ``keywords`` present (case-insensitive) in ``answer``.

    Returns 1.0 when there are no keywords (nothing required).
    """
    if not keywords:
        return 1.0
    lowered = answer.lower()
    return sum(1 for kw in keywords if kw.lower() in lowered) / len(keywords)


def refusal_correct(refused: bool, answerable: bool) -> bool:
    """Answerable questions should be answered; unanswerable ones refused."""
    return refused != answerable
