"""Map Skim Search rows.jsonl records to the diagnostics case contract."""

from __future__ import annotations

import math
from typing import Any, Mapping

_METRICS = (
    "answer_em", "answer_f1", "grounded_em", "grounded_f1",
    "support_f1", "gold_doc_coverage", "set_recall", "tool_errors", "n_steps",
)


def skim_case(row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a scored search_visit episode, preserving its original steps."""
    if row.get("skipped"):
        raise ValueError(f"Skipped episode: {row['skipped']}")
    trajectory = row.get("trajectory")
    if not isinstance(trajectory, list) or "final_answer" not in row:
        raise ValueError("Skim episode requires trajectory and final_answer")
    metrics = {}
    for name in _METRICS:
        value = row.get(name)
        if type(value) in (int, float):
            metrics[name] = value if not isinstance(value, float) or math.isfinite(value) else None
        elif name in row:
            metrics[name] = None
    case = {
        "case_id": row.get("instance_id"),
        "question": row.get("question"),
        "answer": row["final_answer"],
        "status": "error" if row.get("error") else (
            "ok" if row.get("stopped") in ("answer", "submit", "stop") else "unknown"
        ),
        "trajectory": trajectory,
        "metrics": metrics,
    }
    for name in ("stopped", "run_id", "version_id", "error", "traceback"):
        if name in row:
            case[name] = row[name]
    return case
