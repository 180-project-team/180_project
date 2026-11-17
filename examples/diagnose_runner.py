"""Run synthetic scenarios through Runner and Diagnostics without services.

From the repository root: python -m examples.diagnose_runner
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from app.diagnostics import DiagnosisBatchResult, diagnose
from app.runner import AgentRunner, CallableAdapter, RunReport


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
                    "query": "revenue",
                    "observation": "ERROR: Synthetic tool timeout",
                }],
            }
        if question == "repeated_search":
            return {
                "answer": "42",
                "trajectory": [
                    {"action": "bm25_search", "query": "revenue", "observation": "(0 matches)"}
                    for _ in range(2)
                ],
            }
        return {"answer": "42"}


def run_example() -> tuple[RunReport, DiagnosisBatchResult]:
    questions = [{"id": name, "question": name} for name in (
        "normal", "empty", "runtime_error", "tool_error", "repeated_search",
    )]
    run_report = AgentRunner(lambda: CallableAdapter(DemoAgent)).run(questions)
    batch = diagnose(run_report.results, run_id="demo-run", version_id="demo-v1")
    return run_report, batch


def main() -> None:
    run_report, batch = run_example()
    print(json.dumps({"runner": asdict(run_report), "diagnostics": asdict(batch)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
