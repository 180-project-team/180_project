"""Run synthetic scenarios through Runner and Diagnostics without services.

From the repository root: python -m examples.diagnose_runner
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
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


def run_example(*, runs_root: str | Path = "runs") -> tuple[RunReport, DiagnosisBatchResult]:
    questions = [{"id": name, "question": name} for name in (
        "normal", "empty", "runtime_error", "tool_error", "repeated_search",
    )]
    run_report = AgentRunner(lambda: CallableAdapter(DemoAgent), runs_root=runs_root).run(questions)
    batch = diagnose(run_report.load_results(), run_id=run_report.run_id, version_id="demo-v1")
    return run_report, batch


def main() -> None:
    run_report, batch = run_example()
    runner = {**asdict(run_report), "run_dir": str(run_report.run_dir)}
    print(json.dumps({"runner": runner, "diagnostics": asdict(batch)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
