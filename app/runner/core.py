"""Execute a batch using one adapter lifecycle."""

from __future__ import annotations

import json
import time
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Iterable, Iterator, Mapping, TextIO

from .adapters.base import AgentAdapter


@contextmanager
def _open_results(path: Path) -> Iterator[TextIO]:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False)
    temporary_path = Path(output.name)
    try:
        with output:
            yield output
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


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
        results_path: str | Path | None = None,
        continue_on_error: bool = True,
    ) -> Iterator[dict[str, Any]]:
        """Yield rows and publish the supplied JSONL path when the iterator is exhausted."""
        rows = prepare_questions(questions)
        output = _open_results(Path(results_path).absolute()) if results_path is not None else nullcontext()
        with output as stream, ExitStack() as resources:
            if not rows:
                return
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
                if stream is not None:
                    stream.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
                yield result
                if result["status"] == "error" and not continue_on_error:
                    break
