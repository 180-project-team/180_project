"""Demonstrate the in-memory diagnose() interface with AgentRunner.

Run `uv run --locked python -m examples.diagnose_runner` from the repository root.
The demo prints synthetic Runner results and their diagnoses as JSON.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.diagnostics import DiagnosisBatchResult, diagnose
from app.runner import AgentRunner, CallableAdapter


class DemoAgent:
    """Deterministic responses demonstrating observed diagnostic signals."""

    def ask(self, question: str) -> dict[str, Any]:
        if question == "runtime_error":
            raise RuntimeError("Synthetic execution failure")
        if question == "empty":
            return {"answer": ""}
        if question == "tool_error":
            return {
                "answer": "Fallback answer",
                "trajectory": [{
                    "action": "bm25_search",
                    "args": {"query": "revenue"},
                    "observation": "ERROR: Synthetic tool timeout",
                }],
            }
        if question == "repeated_search":
            return {
                "answer": "42",
                "trajectory": [
                    {"action": "bm25_search", "args": {"query": "revenue"}, "observation": "(0 matches)"}
                    for _ in range(2)
                ],
            }
        return {"answer": "42"}


def run_example(*, run_dir: str | Path = ".") -> tuple[list[dict[str, Any]], DiagnosisBatchResult]:
    questions = [{"id": name, "question": name} for name in (
        "normal", "empty", "runtime_error", "tool_error", "repeated_search",
    )]
    runner = AgentRunner(lambda _: CallableAdapter(DemoAgent))
    rows = list(runner.run(questions, run_dir))
    batch = diagnose(rows, run_id="demo-run", version_id="demo-v1")
    return rows, batch


def main() -> None:
    rows, batch = run_example()
    print(json.dumps({"runner": rows, "diagnostics": asdict(batch)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
