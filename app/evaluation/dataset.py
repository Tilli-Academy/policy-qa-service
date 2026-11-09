"""Load and validate the JSONL evaluation dataset."""

from __future__ import annotations

from pathlib import Path

from app.evaluation.schemas import EvalExample

# Project root = app/evaluation/dataset.py -> parents[2]. The dataset lives
# alongside generated reports under evaluation/ at the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "dataset.jsonl"


def load_dataset(path: str | Path | None = None) -> list[EvalExample]:
    """Parse and validate the JSONL dataset.

    Blank lines and ``#``-prefixed comment lines are ignored. Raises
    ``FileNotFoundError`` if the file is missing and ``ValueError`` on an invalid
    record or duplicate id.
    """
    dataset_path = Path(path) if path is not None else DEFAULT_DATASET
    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

    examples: list[EvalExample] = []
    for lineno, raw in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            examples.append(EvalExample.model_validate_json(line))
        except Exception as exc:  # surface the offending line number
            raise ValueError(f"Invalid dataset record at line {lineno}: {exc}") from exc

    ids = [e.id for e in examples]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"Duplicate example ids in dataset: {duplicates}")
    if not examples:
        raise ValueError(f"Evaluation dataset is empty: {dataset_path}")
    return examples
