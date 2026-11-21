"""CLI for a real injected target adapter or a deterministic demo."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any

from app.llm import LLMService
from .evaluation import RunnerEvaluationHarness
from .generator import LLMPatchGenerator
from .models import PromptState
from .optimizer import OptimizationConfig, PromptOptimizer
from .policy import SelectionConfig
from .storage import ExperimentStore


def import_object(path: str) -> Any:
    if ":" not in path:
        raise ValueError("Entry point must use 'module:function' format")
    module_name, name = path.split(":", 1)
    return getattr(importlib.import_module(module_name), name)


def load_questions(path: str) -> list[dict[str, Any]]:
    items = []
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Benchmark rows must be JSON objects")
                items.append(row)
    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Trace-guided prompt optimization")
    parser.add_argument("--prompt-state", required=True, help="PromptState JSON file")
    parser.add_argument("--optimize-questions", required=True, help="Optimization split JSONL")
    parser.add_argument("--validation-questions", required=True, help="Validation split JSONL")
    parser.add_argument("--target", required=True, help="module:function taking (PromptState, seed) -> AgentAdapter")
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--model", help="Model served by an OpenAI-compatible endpoint")
    model.add_argument("--generator", help="module:function returning a patch generator (useful for tests)")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key-env", default="LLM_API_KEY", help="Environment variable holding the LLM endpoint key")
    parser.add_argument("--output", default="runs/prompt_optimizer")
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--candidates-per-iteration", type=int, default=2)
    parser.add_argument("--max-llm-calls", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-delta", type=float, default=0.01)
    parser.add_argument("--max-regressions", type=int, default=0)
    args = parser.parse_args(argv)

    state = PromptState.from_dict(json.loads(Path(args.prompt_state).read_text(encoding="utf-8")))
    optimize_tasks = load_questions(args.optimize_questions)
    validation_tasks = load_questions(args.validation_questions)
    harness = RunnerEvaluationHarness(import_object(args.target))
    config = OptimizationConfig(
        max_iterations=args.max_iterations, candidates_per_iteration=args.candidates_per_iteration,
        max_llm_calls=args.max_llm_calls, seed=args.seed,
        selection=SelectionConfig(min_delta=args.min_delta, max_regressions=args.max_regressions),
    )

    def run(generator: Any) -> None:
        optimizer = PromptOptimizer(harness, generator, store=ExperimentStore(args.output), config=config)
        result = optimizer.optimize(state, optimize_tasks, validation_tasks)
        print(result.session_dir)
        print(f"Selected: {result.selected_version_id}; validation: {result.baseline_validation_score:.3f} -> {result.selected_validation_score:.3f}")

    if args.generator:
        run(import_object(args.generator)())
    else:
        with LLMService(model=args.model, base_url=args.base_url, api_key=os.environ.get(args.api_key_env, "EMPTY")) as llm:
            run(LLMPatchGenerator(llm))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
