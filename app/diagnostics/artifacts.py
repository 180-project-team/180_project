"""Load and join the per-case JSONL artifacts produced by Runner and Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, NoReturn

from .quality import QUALITY_METRICS


_EXECUTION_METRICS = (
    "duration_ms", "tool_calls_count", "successful_tool_calls_count", "failed_tool_calls_count",
)
_JOIN_FIELDS = {"case_id", "evaluation_status", "metrics", "execution_metrics"}


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"{value} is not a JSON value")


def _read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("rb") as source:
        line_number = 0
        try:
            for line_number, raw_line in enumerate(source, 1):
                location = f"{path}:{line_number}"
                try:
                    line = raw_line.decode("utf-8").rstrip("\r\n")
                except UnicodeDecodeError as exc:
                    raise ValueError(f"{location}: invalid UTF-8") from exc
                if not line.strip():
                    continue
                try:
                    row = json.loads(line, parse_constant=_reject_constant)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{location}: invalid JSON at column {exc.colno} ({exc.msg})") from exc
                except ValueError as exc:
                    raise ValueError(f"{location}: invalid JSON ({exc})") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{location}: expected a JSON object")
                yield line_number, row
        except OSError as exc:
            raise OSError(f"{path}:{line_number + 1}: could not read JSONL ({exc})") from exc


def _index_rows(path: Path, *, runner: bool) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    first_lines: dict[str, int] = {}
    for line_number, row in _read_jsonl(path):
        location = f"{path}:{line_number}"
        raw_id = row.get("id")
        if type(raw_id) not in (str, int) or not (case_id := str(raw_id).strip()):
            raise ValueError(f"{location}: expected a non-empty string or integer id")
        if case_id in rows:
            raise ValueError(
                f"{location}: duplicate id {case_id!r} after normalization (first on line {first_lines[case_id]})"
            )
        if row.get("status") not in ("ok", "error"):
            raise ValueError(f"{location}: expected status 'ok' or 'error'")
        if runner:
            question = row.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"{location}: expected a non-empty question string")
            trajectory = row.get("trajectory")
            if trajectory is not None and not isinstance(trajectory, list):
                raise ValueError(f"{location}: expected trajectory to be an array or null")
            if reserved := sorted(_JOIN_FIELDS.intersection(row)):
                raise ValueError(f"{location}: Runner row contains reserved join fields {reserved!r}")
        rows[case_id] = row
        first_lines[case_id] = line_number
    if not rows:
        raise ValueError(f"{path}: expected at least one task")
    return rows


def load_diagnostic_cases(
    run_results_path: str | Path,
    evaluation_results_path: str | Path,
) -> list[dict[str, Any]]:
    """Return one joined record per task, in Runner order.

    Evidence paths are relative to the joined record. Runner fields keep
    their paths except that its normalized id becomes case_id. Evaluator
    status, quality scores and execution metrics have distinct destinations.
    """
    run_path = Path(run_results_path)
    evaluation_path = Path(evaluation_results_path)
    runner_rows = _index_rows(run_path, runner=True)
    evaluation_rows = _index_rows(evaluation_path, runner=False)
    if runner_rows.keys() != evaluation_rows.keys():
        missing_runner = sorted(evaluation_rows.keys() - runner_rows.keys())
        missing_evaluator = sorted(runner_rows.keys() - evaluation_rows.keys())
        raise ValueError(
            f"Task ID sets differ. Missing in Runner ({run_path}): {missing_runner!r}. "
            f"Missing in Evaluator ({evaluation_path}): {missing_evaluator!r}."
        )

    cases = []
    for case_id, runner_row in runner_rows.items():
        evaluation = evaluation_rows[case_id]
        case = {name: value for name, value in runner_row.items() if name != "id"}
        case.update(
            case_id=case_id,
            evaluation_status=evaluation["status"],
            metrics={name: evaluation[name] for name in QUALITY_METRICS if name in evaluation},
            execution_metrics={name: evaluation[name] for name in _EXECUTION_METRICS if name in evaluation},
        )
        cases.append(case)
    return cases
