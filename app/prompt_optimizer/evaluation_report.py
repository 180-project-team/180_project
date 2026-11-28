"""Load existing Evaluator scores; missing measurements remain unknown."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class EvaluationItem:
    task_id: str
    score: float | None
    status: str
    latency_ms: float | None
    tokens: int | None
    trace_ref: str
    error: str | None = None
    trace_events: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.score is not None and (
            type(self.score) not in (int, float)
            or not math.isfinite(self.score)
            or not 0 <= self.score <= 1
        ):
            raise ValueError("Evaluation score must be a number in [0, 1] or None")
        if self.status not in ("ok", "error"):
            raise ValueError("Evaluation status must be ok or error")

    def to_dict(self) -> dict[str, Any]:
        return {**vars(self), "trace_events": list(self.trace_events)}


@dataclass(frozen=True)
class EvaluationReport:
    prompt_version_id: str
    split: str
    run_dir: Path
    items: tuple[EvaluationItem, ...]

    @property
    def score(self) -> float | None:
        if not self.items or any(row.score is None or row.status != "ok" for row in self.items):
            return None
        return sum(row.score for row in self.items) / len(self.items)

    @property
    def scored_tasks(self) -> int:
        return sum(row.score is not None and row.status == "ok" for row in self.items)

    @property
    def total_tokens(self) -> int | None:
        if not self.items or any(row.tokens is None for row in self.items):
            return None
        return sum(row.tokens for row in self.items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_version_id": self.prompt_version_id,
            "split": self.split, "run_dir": str(self.run_dir),
            "score": self.score, "total_tokens": self.total_tokens,
            "scored_tasks": self.scored_tasks, "total_tasks": len(self.items),
            "items": [item.to_dict() for item in self.items],
        }




def _trace_events(row: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = row.get("trajectory", row.get("trace", []))
    if not isinstance(raw, list):
        return ()
    return tuple(dict(event) if isinstance(event, Mapping) else {"event": str(event)} for event in raw)


def _token_count(row: Mapping[str, Any]) -> int | None:
    raw = row.get("token_usage")
    if isinstance(raw, dict):
        raw = raw.get("total_tokens")
        if raw is None:
            return None
    if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
        return raw
    return None

def _load_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or type(row.get("id")) not in (str, int):
                raise ValueError(f"Missing result id in {path}, line {line_number}")
            task_id = str(row["id"]).strip()
            if not task_id or task_id in rows:
                raise ValueError(f"Blank or duplicate result id in {path}, line {line_number}")
            rows[task_id] = row
    return rows

def load_evaluation_report(
    run_results_path: str | Path, evaluation_results_path: str | Path, *,
    prompt_version_id: str, split: str,
) -> EvaluationReport:
    run_path = Path(run_results_path)
    run_rows = _load_by_id(run_path)
    evaluated_rows = _load_by_id(Path(evaluation_results_path))

    if not run_rows or run_rows.keys() != evaluated_rows.keys():
        raise ValueError("Runner and evaluator must contain the same non-empty set of task ids")

    items = []
    for task_id, row in run_rows.items():
        evaluation = evaluated_rows[task_id]
        status = evaluation.get("status")
        if status not in ("ok", "error"):
            raise ValueError(f"Invalid evaluation status for task {task_id}")
        value = evaluation.get("fact_recall")
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"Invalid fact_recall for task {task_id}")
        score = float(value) if value is not None else None
        duration = row.get("duration_ms")

        items.append(EvaluationItem(
            task_id=task_id, score=score, status=status,
            latency_ms=float(duration) if duration is not None else None, tokens=_token_count(row),
            trace_ref=f"{run_path}#{task_id}", error=row.get("error"),
            trace_events=_trace_events(row),
        ))
    return EvaluationReport(prompt_version_id, split, run_path.parent, tuple(items))



