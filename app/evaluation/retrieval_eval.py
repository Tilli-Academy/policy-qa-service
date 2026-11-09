"""Offline retrieval-quality evaluation (no Groq, no LLM).

Runs each answerable dataset question through the real vector index and computes
Hit@k, Recall@k and MRR against the labeled gold chunks. ``run_retrieval_eval``
is pure (returns a report dict, no file I/O); the CLI (``_main``) renders a
Markdown summary and writes timestamped JSON + Markdown reports.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from app.config import settings
from app.evaluation import metrics
from app.evaluation.dataset import PROJECT_ROOT, load_dataset
from app.retrieval.retriever import PolicyRetriever

logger = logging.getLogger(__name__)

REPORTS_DIR = PROJECT_ROOT / "evaluation" / "reports"


def run_retrieval_eval(
    k_values: list[int],
    dataset_path: str | Path | None = None,
    retriever: PolicyRetriever | None = None,
    index_dir: str | None = None,
    embedder=None,
) -> dict:
    """Evaluate retrieval over the answerable examples. Returns a report dict.

    ``retriever`` is injectable for tests; by default the persisted index is
    loaded read-only. Unanswerable examples (no gold chunks) are skipped — they
    belong to the answer/refusal evaluation, not retrieval.
    """
    k_values = sorted(set(k_values))
    max_k = max(k_values)

    examples = [e for e in load_dataset(dataset_path) if e.answerable and e.gold_chunks]
    if retriever is None:
        retriever = PolicyRetriever.from_index(index_dir=index_dir, embedder=embedder)

    items: list[dict] = []
    for ex in examples:
        results = retriever.retrieve(ex.question, top_k=max_k)
        keys = [(r.chunk.source_filename, r.chunk.section_heading) for r in results]
        gold = {(g.source_filename, g.section_heading) for g in ex.gold_chunks}

        gold_scores = [r.score for r in results if (r.chunk.source_filename, r.chunk.section_heading) in gold]
        best_gold_score = max(gold_scores) if gold_scores else None
        items.append(
            {
                "id": ex.id,
                "question": ex.question,
                "tags": ex.tags,
                "hit": {k: metrics.hit_at_k(keys, gold, k) for k in k_values},
                "recall": {k: metrics.recall_at_k(keys, gold, k) for k in k_values},
                "reciprocal_rank": metrics.reciprocal_rank(keys, gold),
                "first_hit_rank": metrics.first_hit_rank(keys, gold),
                "best_gold_score": best_gold_score,
                "below_floor": (best_gold_score is None) or (best_gold_score < settings.answer_min_score),
            }
        )

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "index_dir": index_dir if index_dir is not None else settings.index_dir,
        "k_values": k_values,
        "num_examples": len(items),
        "min_score_threshold": settings.answer_min_score,
        "embedding_model": settings.embedding_model,
        "hit_at_k": {k: metrics._mean(i["hit"][k] for i in items) for k in k_values},
        "recall_at_k": {k: metrics._mean(i["recall"][k] for i in items) for k in k_values},
        "mrr": metrics._mean(i["reciprocal_rank"] for i in items),
        "num_below_floor": sum(1 for i in items if i["below_floor"]),
        "items": items,
    }
    return report


def render_markdown(report: dict) -> str:
    """Render a human-readable Markdown summary of a retrieval report."""
    ks = report["k_values"]
    lines = [
        f"# Retrieval Evaluation — {report['timestamp']}",
        "",
        f"- Index: `{report.get('index_dir')}`",
        f"- Examples (answerable): **{report['num_examples']}**",
        f"- Embedding model: `{report['embedding_model']}`",
        f"- ANSWER_MIN_SCORE: `{report['min_score_threshold']}`  |  gold below floor: "
        f"**{report['num_below_floor']}**",
        "",
        "## Aggregate",
        "",
        "| Metric | " + " | ".join(f"k={k}" for k in ks) + " |",
        "|" + "---|" * (len(ks) + 1),
        "| Hit@k | " + " | ".join(f"{report['hit_at_k'][k]:.3f}" for k in ks) + " |",
        "| Recall@k | " + " | ".join(f"{report['recall_at_k'][k]:.3f}" for k in ks) + " |",
        "",
        f"**MRR = {report['mrr']:.3f}**",
        "",
        "## Per-item",
        "",
        "| id | first_hit_rank | best_gold_score | below_floor |",
        "|---|---|---|---|",
    ]
    for i in report["items"]:
        rank = i["first_hit_rank"] if i["first_hit_rank"] is not None else "—"
        score = f"{i['best_gold_score']:.4f}" if i["best_gold_score"] is not None else "—"
        lines.append(f"| {i['id']} | {rank} | {score} | {'yes' if i['below_floor'] else 'no'} |")
    return "\n".join(lines) + "\n"


def write_reports(report: dict, markdown: str, kind: str = "retrieval") -> tuple[Path, Path]:
    """Write timestamped JSON + Markdown reports; return their paths."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = REPORTS_DIR / f"{kind}_{stamp}.json"
    md_path = REPORTS_DIR / f"{kind}_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    return json_path, md_path


def _main() -> None:
    parser = argparse.ArgumentParser(description="Offline retrieval-quality evaluation (no Groq).")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5], help="k values for @k metrics.")
    parser.add_argument("--dataset", default=None, help="Path to a JSONL dataset (defaults to evaluation/dataset.jsonl).")
    parser.add_argument("--index-dir", default=None, help="Index directory (defaults to settings.index_dir).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    report = run_retrieval_eval(args.k, dataset_path=args.dataset, index_dir=args.index_dir)
    markdown = render_markdown(report)
    json_path, md_path = write_reports(report, markdown, kind="retrieval")
    print(markdown)
    print(f"Reports written:\n  {json_path}\n  {md_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    _main()
