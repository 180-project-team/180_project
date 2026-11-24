"""Execute a batch using one adapter lifecycle."""

from __future__ import annotations

import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

from .adapters.base import AgentAdapter


def prepare_questions(questions: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    batch = [dict(row) if isinstance(row, Mapping) else row for row in questions]
    for index, row in enumerate(batch, 1):
        if not isinstance(row, dict) or not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError(f"Question #{index} must contain a non-empty 'question' string")
    return batch


class AgentRunner:
    def __init__(self, adapter_factory: Callable[[Path], AgentAdapter]) -> None:
        if not callable(adapter_factory):
            raise TypeError("adapter_factory must be callable")
        self._factory = adapter_factory

    def run(
        self,
        questions: Iterable[Mapping[str, Any]],
        run_dir: str | Path,
        *,
        continue_on_error: bool = True,
    ) -> Iterator[dict[str, Any]]:
        """Yield each result; run_dir is the directory supplied by the caller."""
        rows = prepare_questions(questions)
        if not rows:
            return
        with ExitStack() as resources:
            adapter = self._factory(Path(run_dir))
            resources.callback(adapter.stop)
            adapter.start()
            for index, row in enumerate(rows, 1):
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
                result["duration_ms"] = round((time.monotonic() - begin) * 1000, 2)
                yield result
                if result["status"] == "error" and not continue_on_error:
                    break
