"""Small orchestration example (the production orchestrator is a separate task).

Run without a model: uv run python -m examples.orchestrator_demo
With a real model, pass llm_config to ExampleOrchestrator, which constructs
and owns a shared LLMService for its runner/evaluator/optimizer workflow.
"""

from pathlib import Path
import sys
from typing import Any

from app.llm import LLMService
from app.runner import AgentRunner, SubprocessJSONLAdapter, load_questions


class ExampleOrchestrator:
    def __init__(self, runner: AgentRunner, *, llm_config: dict[str, Any] | None = None):
        self.runner = runner
        self.llm = LLMService(**llm_config) if llm_config is not None else None

    def __enter__(self):
        if self.llm is not None:
            self.llm.start()
        return self

    def __exit__(self, *_):
        if self.llm is not None:
            self.llm.stop()

    def execute(self, questions):
        return self.runner.run(questions, metadata={"created_by": "example_orchestrator"})


if __name__ == "__main__":
    agent = Path(__file__).resolve().parent / "echo_agent.py"
    runner = AgentRunner(lambda: SubprocessJSONLAdapter([sys.executable, "-u", str(agent)]))
    with ExampleOrchestrator(runner) as orchestrator:
        report = orchestrator.execute(load_questions(agent.parent / "questions.jsonl"))
        print(f"Artifacts: {report.run_dir}")
