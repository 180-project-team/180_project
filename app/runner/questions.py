"""Inputs accepted by the batch runner: JSONL, JSON, or Hugging Face dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _validate(item: Any, number: int) -> dict[str, Any]:
    if isinstance(item, str):
        item = {"id": str(number), "question": item}
    if not isinstance(item, dict) or not isinstance(item.get("question"), str) or not item["question"].strip():
        raise ValueError(f"Question #{number} needs a nonempty 'question' string")
    return item


def load_questions(path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Read lines such as {"id":"1","question":"...","answers":["..."]}."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if path.suffix.lower() == ".jsonl":
        items: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as source:
            for line_no, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_no} of {path}") from exc
                items.append(_validate(row, line_no))
                if limit is not None and len(items) >= limit:
                    break
        return items
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as source:
            data = json.load(source)
        if isinstance(data, dict):
            data = data.get("questions")
        if not isinstance(data, list):
            raise ValueError("Expected a JSON list or {'questions': [...]} object")
        return [_validate(row, n) for n, row in enumerate(data[:limit] if limit else data, 1)]
    raise ValueError("Questions file must end with .json or .jsonl")


def load_hf_questions(dataset_name: str, split: str, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Optional dependency; loads MADQA or compatible HF datasets on demand."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install 'datasets' to use --dataset") from exc
    dataset = load_dataset(dataset_name, split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return [_validate(dict(row), i) for i, row in enumerate(dataset, 1)]
