from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .metrics import METRIC_NAMES


EXECUTION_METRIC_NAMES = (
    "duration_ms",
    "tool_calls_count",
    "successful_tool_calls_count",
    "failed_tool_calls_count",
)

RESULT_COLUMNS = (
    "id",
    "status",
    *METRIC_NAMES,
    *EXECUTION_METRIC_NAMES,
)


def _numeric_values(
    rows: Sequence[dict[str, Any]],
    name: str,
    *,
    successful_only: bool = False,
) -> list[float]:
    values: list[float] = []

    for row in rows:
        if successful_only and row.get("status") != "ok":
            continue

        value = row.get(name)
        if value is None or isinstance(value, bool):
            continue

        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue

    return values


def _mean_and_count(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "mean": sum(values) / len(values) if values else None,
        "count": len(values),
    }


def build_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total_questions": len(rows),
        "successful_questions": sum(row.get("status") == "ok" for row in rows),
        "failed_questions": sum(row.get("status") != "ok" for row in rows),
    }

    for name in METRIC_NAMES:
        summary[name] = _mean_and_count(_numeric_values(rows, name))

    for name in EXECUTION_METRIC_NAMES:
        summary[name] = _mean_and_count(
            _numeric_values(rows, name, successful_only=True)
        )

    return summary


def write_results(output_dir: Path, rows: Sequence[dict[str, Any]]) -> None:
    jsonl_path = output_dir / "results.jsonl"
    csv_path = output_dir / "results.csv"

    with jsonl_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(
            {name: row.get(name) for name in RESULT_COLUMNS} for row in rows
        )


def _flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "total_questions": summary["total_questions"],
        "successful_questions": summary["successful_questions"],
        "failed_questions": summary["failed_questions"],
    }

    for name in (*METRIC_NAMES, *EXECUTION_METRIC_NAMES):
        data = summary[name]
        row[f"{name}_mean"] = data["mean"]
        row[f"{name}_count"] = data["count"]

    return row


def write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"

    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    row = _flatten_summary(summary)
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_evaluation_report(
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
) -> None:
    summary = build_summary(rows)
    write_results(output_dir, rows)
    write_summary(output_dir, summary)
