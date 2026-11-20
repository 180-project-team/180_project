"""Agent-agnostic execution engine. The orchestrator owns the runner.

Every invocation produces a new directory containing questions, responses,
execution events, a summary, and a space for extra diagnostic artifacts.
There is deliberately NO dependency on evaluators or LLM services here.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .adapters.base import AgentAdapter


@dataclass(frozen=True)
class RunReport:
    """Location of persisted run artifacts, not an in-memory result payload."""

    run_id: str
    run_dir: Path
    processed_count: int
    failed_count: int
    status: str

    @property
    def results_path(self) -> Path:
        return self.run_dir / "results.jsonl"

    @property
    def questions_path(self) -> Path:
        return self.run_dir / "questions.jsonl"

    @property
    def metadata_path(self) -> Path:
        return self.run_dir / "metadata.json"

    @property
    def events_path(self) -> Path:
        return self.run_dir / "events.jsonl"

    def load_results(self) -> list[dict[str, Any]]:
        """Optional convenience method; evaluators may stream results.jsonl instead."""
        with self.results_path.open(encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]


class AgentRunner:
    """Run a batch with one agent process lifecycle.

    The orchestrator creates an AgentRunner and passes questions to run().
    Each call creates a unique directory under runs_root. The adapter is
    constructed afresh for each call and stopped even on startup failure.
    An evaluator can depend on this class; this class never imports one.
    """

    def __init__(
        self,
        adapter_factory: Callable[[], AgentAdapter],
        *,
        runs_root: str | Path = "runs",
    ) -> None:
        if not callable(adapter_factory):
            raise TypeError("adapter_factory must be callable")
        self._factory = adapter_factory
        self.runs_root = Path(runs_root).expanduser().resolve()

    def run(
        self,
        questions: Iterable[Mapping[str, Any]],
        *,
        run_id: str | None = None,
        continue_on_error: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> RunReport:
        batch = [dict(row) if isinstance(row, Mapping) else row for row in questions]
        for index, row in enumerate(batch, 1):
            if not isinstance(row, dict) or not isinstance(row.get("question"), str) or not row["question"].strip():
                raise ValueError(f"Question #{index} must contain a non-empty 'question' string")

        identifier = run_id or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8])
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", identifier) or identifier in {".", ".."}:
            raise ValueError("run_id must be a simple filename (letters, numbers, _, -, .)")

        self.runs_root.mkdir(parents=True, exist_ok=True)
        run_dir = self.runs_root / identifier
        run_dir.mkdir(exist_ok=False)  # Never silently overwrite a previous run.
        (run_dir / "artifacts").mkdir()
        questions_file = run_dir / "questions.jsonl"
        results_file = run_dir / "results.jsonl"
        events_file = run_dir / "events.jsonl"
        metadata_file = run_dir / "metadata.json"

        def write_json(path: Path, value: Any) -> None:
            # Atomic summary writes prevent partially written metadata files.
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
            temporary.replace(path)

        with questions_file.open("w", encoding="utf-8") as file:
            for row in batch:
                file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

        summary: dict[str, Any] = {
            "run_id": identifier,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "question_count": len(batch),
            "processed_count": 0,
            "failed_count": 0,
            "metadata": dict(metadata) if metadata else {},
        }
        write_json(metadata_file, summary)

        def event(file: Any, name: str, **fields: Any) -> None:
            file.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "event": name, **fields}, ensure_ascii=False, default=str) + "\n")
            file.flush()

        try:
            with events_file.open("w", encoding="utf-8") as events, results_file.open("w", encoding="utf-8") as sink:
                event(events, "run_started", question_count=len(batch))
                if batch:
                    with ExitStack() as resources:
                        adapter = self._factory()
                        resources.callback(adapter.stop)  # Cleanup even if start() fails.
                        set_artifacts_dir = getattr(adapter, "set_artifacts_dir", None)
                        if callable(set_artifacts_dir):
                            set_artifacts_dir(run_dir / "artifacts")
                        adapter.start()
                        event(events, "agent_started")
                        for index, row in enumerate(batch, 1):
                            begin = time.monotonic()
                            result: dict[str, Any] = dict(row)
                            result["id"] = str(row.get("id") if row.get("id") is not None else index)
                            result["question"] = row["question"]
                            try:
                                raw = adapter.ask(row["question"])
                                if isinstance(raw, Mapping):
                                    if raw.get("error") is not None:
                                        raise RuntimeError(str(raw["error"]))
                                    if "answer" not in raw:
                                        raise ValueError("Agent response must contain 'answer'")
                                    # Keep reference answers, question, run metadata intact.
                                    result.update({k: v for k, v in raw.items() if k not in row and k not in {"id", "question", "status", "duration_ms"}})
                                    result["answer"] = raw["answer"]
                                else:
                                    result["answer"] = raw
                                result["status"] = "ok"
                            except Exception as exc:
                                result["answer"] = None
                                result["status"] = "error"
                                result["error"] = f"{type(exc).__name__}: {exc}"
                                event(events, "question_failed", question_id=result["id"], error=result["error"])
                            result["duration_ms"] = round((time.monotonic() - begin) * 1000, 2)
                            sink.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
                            sink.flush()
                            summary["processed_count"] += 1
                            summary["failed_count"] += result["status"] == "error"
                            if result["status"] == "error" and not continue_on_error:
                                break
                    event(events, "agent_stopped")
                summary["status"] = "completed_with_errors" if summary["failed_count"] else "completed"
                event(events, "run_finished", status=summary["status"])
        except BaseException as exc:
            summary["status"] = "failed"
            summary["fatal_error"] = f"{type(exc).__name__}: {exc}"
            # The run directory remains available for debugging after exceptions.
            with events_file.open("a", encoding="utf-8") as events:
                event(events, "run_failed", error=summary["fatal_error"])
            raise
        finally:
            summary["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_json(metadata_file, summary)

        return RunReport(identifier, run_dir, summary["processed_count"], summary["failed_count"], summary["status"])
