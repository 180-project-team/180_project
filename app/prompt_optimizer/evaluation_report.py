"""Evaluate an immutable prompt with the project's existing AgentRunner."""

from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class EvaluationItem:
    task_id: str
    score: float
    status: str
    latency_ms: float
    tokens: int | None
    trace_ref: str
    error: str | None = None
    trace_events: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {**vars(self), "trace_events": list(self.trace_events)}


@dataclass(frozen=True)
class EvaluationReport:
    prompt_version_id: str
    split: str
    run_dir: Path
    items: tuple[EvaluationItem, ...]

    @property
    def score(self) -> float:
        return sum(row.score for row in self.items) / len(self.items) if self.items else 0.0

    @property
    def total_tokens(self) -> int | None:
        return sum(row.tokens for row in self.items if row.tokens is not None) if all(row.tokens is not None for row in self.items) else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_version_id": self.prompt_version_id,
            "split": self.split, "run_dir": str(self.run_dir),
            "score": self.score, "total_tokens": self.total_tokens,
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
            if not isinstance(row, dict) or row.get("id") is None:
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
        if status == "error":
            score = 0.0
        else:
            value = evaluation.get("fact_recall")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"Missing or invalid fact_recall for task {task_id}")
            score = float(value)

        items.append(EvaluationItem(
            task_id=task_id, score=score, status=status,
            latency_ms=float(row.get("duration_ms", 0)), tokens=_token_count(row),
            trace_ref=f"{run_path}#{task_id}", error=row.get("error"),
            trace_events=_trace_events(row),
        ))
    return EvaluationReport(prompt_version_id, split, run_path.parent, tuple(items))



