"""Evaluate an immutable prompt with the project's existing AgentRunner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from app.runner import AgentRunner
from .models import PromptState
from .renderer import PromptRenderer


@dataclass(frozen=True)
class EvaluationItem:
    task_id: str
    score: float
    status: str
    latency_ms: float
    tokens: int | None
    trace_ref: str
    error: str | None = None
    trace_events: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {**vars(self), "trace_events": list(self.trace_events)}


@dataclass(frozen=True)
class EvaluationReport:
    prompt_version_id: str
    split: str
    run_dir: Path
    items: tuple[EvaluationItem, ...]

    @property
    def score(self) -> float:
        return sum(row.score for row in self.items) / len(self.items) if self.items else 0.0

    @property
    def total_tokens(self) -> int | None:
        return sum(row.tokens for row in self.items if row.tokens is not None) if all(row.tokens is not None for row in self.items) else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_version_id": self.prompt_version_id,
            "split": self.split, "run_dir": str(self.run_dir),
            "score": self.score, "total_tokens": self.total_tokens,
            "items": [item.to_dict() for item in self.items],
        }


class TargetFactory(Protocol):
    def __call__(self, state: PromptState, seed: int) -> Any: ...


class EvaluationHarness(Protocol):
    def evaluate(
        self, state: PromptState, tasks: Sequence[Mapping[str, Any]], *,
        seed: int, split: str, runs_root: Path,
    ) -> EvaluationReport: ...


def _normalized(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(_normalized(item) for item in value).strip()
    return " ".join(str(value).casefold().split())


def exact_match(row: Mapping[str, Any]) -> float:
    """A small baseline scorer for datasets with known reference answers.

    Production evaluations can inject their own scorer through RunnerEvaluationHarness.
    """
    if row.get("status") != "ok":
        return 0.0
    references = row.get("answer_variants")
    if references is None:
        references = row.get("reference_answers")
    if references is None:
        references = [row["reference_answer"]] if "reference_answer" in row else []
    if not isinstance(references, list):
        raise ValueError("Expected a list of reference answers")
    prediction = _normalized(row.get("answer", ""))
    variants = [item for group in references for item in (group if isinstance(group, list) else [group])]
    return float(bool(prediction) and any(prediction == _normalized(reference) for reference in variants))


def _trace_events(row: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = row.get("trajectory", row.get("trace", []))
    if not isinstance(raw, list):
        return ()
    return tuple(dict(event) if isinstance(event, Mapping) else {"event": str(event)} for event in raw)


def _token_count(row: Mapping[str, Any]) -> int | None:
    raw = row.get("token_usage")
    if isinstance(raw, dict):
        raw = raw.get("total_tokens")
        if raw is None:
            return None
    if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
        return raw
    return None


class RunnerEvaluationHarness:
    """Runner remains independent of evaluator and prompt optimizer.

    The supplied factory is responsible for binding PromptState to its target
    agent; this module never alters the target agent's code.
    """

    def __init__(
        self, target_factory: TargetFactory, *,
        scorer: Callable[[Mapping[str, Any]], float] = exact_match,
        renderer: PromptRenderer | None = None,
    ) -> None:
        if not callable(target_factory) or not callable(scorer):
            raise TypeError("target_factory and scorer must be callable")
        self.target_factory = target_factory
        self.scorer = scorer
        self.renderer = renderer or PromptRenderer()

    def evaluate(
        self, state: PromptState, tasks: Sequence[Mapping[str, Any]], *,
        seed: int, split: str, runs_root: Path,
    ) -> EvaluationReport:
        self.renderer.validate_bindable(state)
        if not tasks:
            raise ValueError("Evaluation requires non-empty tasks")
        prepared: list[dict[str, Any]] = []
        for index, task in enumerate(tasks, 1):
            row = dict(task)
            row["id"] = str(row.get("id", index))
            if not row.get("question") or not isinstance(row["question"], str):
                raise ValueError("Each task must contain a non-empty question")
            # Runner overwrites answer with prediction; keep dataset labels separate.
            if "answer" in row and "reference_answer" not in row and "answer_variants" not in row:
                row["reference_answer"] = row.pop("answer")
            if self.scorer is exact_match and not any(name in row for name in ("answer_variants", "reference_answers", "reference_answer")):
                raise ValueError(f"Missing reference answer for task {row['id']}")
            prepared.append(row)
        if len({row["id"] for row in prepared}) != len(prepared):
            raise ValueError("Task identifiers must be unique within a benchmark split")
        runner = AgentRunner(lambda: self.target_factory(state, seed), runs_root=runs_root)
        result = runner.run(prepared, metadata={
            "prompt_version_id": state.prompt_version_id, "seed": seed, "split": split,
        })
        items = []
        for row in result.load_results():
            score = float(self.scorer(row))
            if not 0.0 <= score <= 1.0:
                raise ValueError("Evaluator scores must be in [0, 1]")
            items.append(EvaluationItem(
                task_id=str(row["id"]), score=score, status=row["status"],
                latency_ms=float(row.get("duration_ms", 0)), tokens=_token_count(row),
                trace_ref=f"{result.results_path}#{row['id']}", error=row.get("error"),
                trace_events=_trace_events(row),
            ))
        if len(items) != len(prepared):
            raise RuntimeError("Evaluation did not process the entire benchmark")
        return EvaluationReport(state.prompt_version_id, split, result.run_dir, tuple(items))
