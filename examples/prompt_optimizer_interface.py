"""Offline interface demo using shared Diagnostics and a fixed patch generator.

Run: uv run --locked python -m examples.prompt_optimizer_interface
This checks artifact exchange; it does not measure a quality improvement.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from app.diagnostics import diagnose
from app.prompt_optimizer import PromptOptimizer

BEFORE = 'SYSTEM_PROMPT = "Answer {question}."'
AFTER = 'SYSTEM_PROMPT = "Use the available evidence and return a clear final answer to {question}."'
AGENT_CODE = (
    "agent/\n└── main.py\n\n# main.py\n" + BEFORE
    + "\n\ndef make_prompt(question):\n    return SYSTEM_PROMPT.format(question=question)\n"
)


class DemoSourceGenerator:
    """A deterministic proposal for this demo's source, not a quality model."""

    def generate(self, agent_code, reports):
        report = reports[0]
        return {
            "report_index": report["report_index"],
            "hypothesis_id": report["hypotheses"][0]["hypothesis_id"],
            "before": BEFORE, "after": AFTER,
            "rationale": "Make the instruction to return a grounded final answer explicit.",
        }


def make_generator():
    return DemoSourceGenerator()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check the prompt optimizer's file-based interface offline")
    parser.add_argument("--output-dir", "--output", dest="output_dir", type=Path, default=Path("runs/prompt_optimizer_interface"))
    args = parser.parse_args(argv)

    inputs = args.output_dir / "input"
    inputs.mkdir(parents=True, exist_ok=True)
    # Input preparation is outside PromptOptimizer and uses the team's Diagnostics.
    batch = diagnose([{
        "id": "demo-1", "question": "Who directed both films?", "answer": None,
        "trajectory": [], "status": "ok",
    }], run_id="demo-run", version_id="agent-v1")
    diagnosis_path = inputs / "diagnoses.jsonl"
    diagnosis_path.write_text("".join(json.dumps(asdict(report)) + "\n" for report in batch.reports), encoding="utf-8")
    (inputs / "agent_code.txt").write_text(AGENT_CODE, encoding="utf-8")

    result = PromptOptimizer(generator=make_generator()).optimize(
        diagnosis_path, args.output_dir, AGENT_CODE,
    )
    print(f"Offline interface demo: {result.status}")
    print(result.result_path)
    print(result.reason)
    return 0 if result.status == "proposed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
