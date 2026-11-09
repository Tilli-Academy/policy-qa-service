"""Answer-quality evaluation (one Groq call per answerable item; opt-in).

For each dataset item this runs the real QA pipeline (``answer_question``) and
scores the result deterministically — no LLM-as-judge:

  * refusal correctness (answerable -> answered, unanswerable -> refused),
  * citation correctness (a returned citation matches a gold chunk),
  * keyword recall against ``answer_must_contain``.

``run_answer_eval`` is pure of file I/O and accepts an injected retriever/LLM so
tests exercise it offline with fakes. The CLI (``_main``) constructs the live
``GroqLLM`` (reading GROQ_API_KEY from the environment — never logged or printed)
and makes real calls; it is intended to be run deliberately.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from app.config import settings
from app.evaluation import metrics
from app.evaluation.dataset import load_dataset
from app.evaluation.retrieval_eval import REPORTS_DIR
from app.llm.base import BaseLLM
from app.retrieval.retriever import PolicyRetriever
from app.services.qa import answer_question

logger = logging.getLogger(__name__)


class _CountingLLM(BaseLLM):
    """Wraps an LLM to count how many times it is actually invoked."""

    def __init__(self, inner: BaseLLM) -> None:
        self._inner = inner
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        return self._inner.generate(system_prompt, user_prompt)


def run_answer_eval(
    dataset_path: str | Path | None = None,
    retriever: PolicyRetriever | None = None,
    llm: BaseLLM | None = None,
    top_k: int | None = None,
    min_score: float | None = None,
) -> dict:
    """Run the answer pipeline over the dataset and score it. Returns a report dict."""
    examples = load_dataset(dataset_path)
    if retriever is None:
        retriever = PolicyRetriever.from_index()
    if llm is None:  # pragma: no cover - live path, exercised only via CLI
        from app.llm.groq_client import GroqLLM

        llm = GroqLLM()
    counter = _CountingLLM(llm)

    items: list[dict] = []
    for ex in examples:
        resp = answer_question(
            ex.question, retriever=retriever, llm=counter, top_k=top_k, min_score=min_score
        )
        gold = {(g.source_filename, g.section_heading) for g in ex.gold_chunks}
        citation_keys = [(c.source_filename, c.section_heading) for c in resp.citations]
        kw_recall = (
            metrics.keyword_recall(resp.answer, ex.answer_must_contain)
            if (ex.answerable and not resp.refused)
            else None
        )
        items.append(
            {
                "id": ex.id,
                "answerable": ex.answerable,
                "refused": resp.refused,
                "refusal_correct": metrics.refusal_correct(resp.refused, ex.answerable),
                "citation_hit": metrics.citation_hit(citation_keys, gold) if ex.answerable else None,
                "keyword_recall": kw_recall,
                "answer": resp.answer,
            }
        )

    answerable = [i for i in items if i["answerable"]]
    answered = [i for i in answerable if not i["refused"]]
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "num_examples": len(items),
        "num_answerable": len(answerable),
        "groq_model": settings.groq_model,
        "groq_calls": counter.calls,
        "refusal_accuracy": metrics._mean(i["refusal_correct"] for i in items),
        "citation_accuracy": metrics._mean(i["citation_hit"] for i in answerable),
        "keyword_pass_rate": metrics._mean((i["keyword_recall"] == 1.0) for i in answered),
        "keyword_mean_recall": metrics._mean(i["keyword_recall"] for i in answered),
        "items": items,
    }
    return report


def render_markdown(report: dict) -> str:
    """Render a Markdown summary of an answer-eval report."""
    lines = [
        f"# Answer Evaluation — {report['timestamp']}",
        "",
        f"- Model: `{report['groq_model']}`  |  Groq calls: **{report['groq_calls']}**",
        f"- Examples: **{report['num_examples']}** (answerable: {report['num_answerable']})",
        "",
        "## Aggregate",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Refusal accuracy (all) | {report['refusal_accuracy']:.3f} |",
        f"| Citation accuracy (answerable) | {report['citation_accuracy']:.3f} |",
        f"| Keyword pass-rate (answered) | {report['keyword_pass_rate']:.3f} |",
        f"| Keyword mean recall (answered) | {report['keyword_mean_recall']:.3f} |",
        "",
        "## Per-item",
        "",
        "| id | answerable | refused | refusal_ok | citation_hit | keyword_recall |",
        "|---|---|---|---|---|---|",
    ]
    for i in report["items"]:
        ch = "—" if i["citation_hit"] is None else ("yes" if i["citation_hit"] else "no")
        kw = "—" if i["keyword_recall"] is None else f"{i['keyword_recall']:.2f}"
        lines.append(
            f"| {i['id']} | {'yes' if i['answerable'] else 'no'} | "
            f"{'yes' if i['refused'] else 'no'} | {'yes' if i['refusal_correct'] else 'no'} | "
            f"{ch} | {kw} |"
        )
    return "\n".join(lines) + "\n"


def write_reports(report: dict, markdown: str) -> tuple[Path, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = REPORTS_DIR / f"answer_{stamp}.json"
    md_path = REPORTS_DIR / f"answer_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    return json_path, md_path


def _main() -> None:  # pragma: no cover - CLI entry point (makes live Groq calls)
    parser = argparse.ArgumentParser(
        description="Answer-quality evaluation. Makes ~1 live Groq call per answerable item."
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    examples = load_dataset(args.dataset)
    answerable = sum(1 for e in examples if e.answerable)
    print(
        f"This will make up to ~{answerable} live Groq calls (model: {settings.groq_model}).\n"
        "No API key is printed. Retrieval-only evaluation is free via retrieval_eval."
    )
    if not args.yes:
        reply = input("Proceed with live answer evaluation? [y/N] ").strip().lower()
        if reply not in {"y", "yes"}:
            print("Aborted.")
            return

    report = run_answer_eval(dataset_path=args.dataset)
    markdown = render_markdown(report)
    json_path, md_path = write_reports(report, markdown)
    print(markdown)
    print(f"Reports written:\n  {json_path}\n  {md_path}")


if __name__ == "__main__":
    _main()
