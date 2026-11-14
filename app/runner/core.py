"""Agent-agnostic batch runner. Evaluation is supplied by the caller."""

from __future__ import annotations

import json
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol

from .adapters.base import AgentAdapter


class Evaluator(Protocol):
    def __call__(self, results: list[dict[str, Any]]) -> Any: ...


@dataclass
class RunReport:
    """Answers and optional *external* evaluator output."""

    results: list[dict[str, Any]]
    evaluation: Any = None

    @property
    def failed_count(self) -> int:
        return sum(row["status"] == "error" for row in self.results)


class AgentRunner:
    """Run N questions with ONE agent lifecycle; always stop resources we own.

    `adapter_factory` must return a NEW adapter for each run. This avoids reusing
    stopped subprocesses and permits comparing agents or prompt variants.

    `model_service` is an optional context manager (e.g. ManagedModelServer).
    It starts before the adapter and is stopped after the adapter.

    Evaluator receives all normalized rows, including failures. No scoring is
    performed in this module.
    """

    def __init__(
        self,
        adapter_factory: Callable[[], AgentAdapter],
        *,
        model_service: Any = None,
    ) -> None:
        if not callable(adapter_factory):
            raise TypeError("adapter_factory must be callable and create a new adapter")
        self._factory = adapter_factory
        self._model_service = model_service

    def run(
        self,
        questions: Iterable[Mapping[str, Any]],
        *,
        evaluator: Evaluator | None = None,
        output_path: str | Path | None = None,
        continue_on_error: bool = True,
    ) -> RunReport:
        rows = list(questions)
        for i, row in enumerate(rows, 1):
            if (
                not isinstance(row, Mapping)
                or not isinstance(row.get("question"), str)
                or not row["question"].strip()
            ):
                raise ValueError(f"Question #{i} must contain a non-empty 'question' string")

        # No expensive startup if no work. Do not overwrite existing output.
        if not rows:
            return RunReport([], evaluator([]) if evaluator else None)

        destination = Path(output_path) if output_path is not None else None
        if destination is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)

        results: list[dict[str, Any]] = []
        with ExitStack() as resources:
            if self._model_service is not None:
                resources.enter_context(self._model_service)

            adapter = self._factory()
            # Register cleanup BEFORE start: partly initialized adapters can fail.
            resources.callback(adapter.stop)
            adapter.start()

            sink = resources.enter_context(destination.open("w", encoding="utf-8")) if destination else None
            for i, row in enumerate(rows, 1):
                begin = time.monotonic()
                # Copy input fields: preserves ground truth (MADQA: answer_variants/evidence)
                # without requiring the runner to understand a specific benchmark.
                result: dict[str, Any] = dict(row)
                result["id"] = str(row.get("id") if row.get("id") is not None else i)
                result["question"] = row["question"]
                try:
                    raw = adapter.ask(row["question"])
                    if isinstance(raw, Mapping):
                        if "error" in raw:
                            raise RuntimeError(str(raw["error"]))
                        # Prevent agent outputs from overriding the task identity/GT.
                        result.update({k: v for k, v in raw.items() if k not in row and k not in {"id", "question", "status", "duration_ms"}})
                        if "answer" not in raw:
                            raise ValueError("Adapter response dictionary must contain 'answer'")
                        result["answer"] = raw["answer"]
                    else:
                        result["answer"] = raw
                    result["status"] = "ok"
                except Exception as exc:
                    result["answer"] = None
                    result["status"] = "error"
                    result["error"] = f"{type(exc).__name__}: {exc}"
                    # A broken adapter may be unusable after an exception, so
                    # continue_on_error is intended for per-question failures.
                result["duration_ms"] = round((time.monotonic() - begin) * 1000, 2)
                results.append(result)
                if sink is not None:
                    sink.write(json.dumps(result, ensure_ascii=False) + "\n")
                    sink.flush()
                if result["status"] == "error" and not continue_on_error:
                    break

            # Evaluator runs BEFORE agent shutdown (allows callbacks to inspect it).
            # A raised evaluator error still triggers resource cleanup via ExitStack.
            evaluation = evaluator(results) if evaluator else None
            return RunReport(results, evaluation)
