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

    def execute(self, questions, run_dir: str | Path, *, results_path: str | Path | None = None):
        destination = Path(run_dir) / "runner_results.jsonl" if results_path is None else Path(results_path)
        return list(self.runner.run(questions, run_dir, results_path=destination))


if __name__ == "__main__":
    agent = Path(__file__).resolve().parent / "echo_agent.py"
    runner = AgentRunner(lambda _: SubprocessJSONLAdapter([sys.executable, "-u", str(agent)]))
    with ExampleOrchestrator(runner) as orchestrator:
        for row in orchestrator.execute(load_questions(agent.parent / "questions.jsonl"), Path("runs/demo")):
            print(row)
