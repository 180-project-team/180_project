"""Baseline -> diagnosis -> patch -> re-evaluation -> measured selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .diagnosis import BehaviorAnalyzer, LessonExtractor, PromptAttribution
from .evaluation import EvaluationHarness, EvaluationReport
from .generator import PromptModelPort
from .models import PromptState
from .policy import PromptGuard, SelectionConfig, SelectionPolicy
from .storage import ExperimentStore, write_json


@dataclass(frozen=True)
class OptimizationConfig:
    max_iterations: int = 3
    candidates_per_iteration: int = 2
    max_llm_calls: int = 12
    seed: int = 42
    selection: SelectionConfig = SelectionConfig()

    def __post_init__(self) -> None:
        if self.max_iterations < 1 or self.candidates_per_iteration < 1 or self.max_llm_calls < 1:
            raise ValueError("Optimization budgets must be positive")


@dataclass(frozen=True)
class OptimizationSessionResult:
    session_dir: Path
    baseline_version_id: str
    selected_version_id: str
    accepted_iterations: int
    stop_reason: str
    baseline_validation_score: float
    selected_validation_score: float


class PromptOptimizer:
    """Coordinates optimization without owning the runner, target agent or LLM.

    Generation sees optimize-split failures only. Selection compares candidate
    and parent on a separate validation split using identical seeds and tasks.
    Final test data must never be passed to this object.
    """

    def __init__(
        self, harness: EvaluationHarness, generator: PromptModelPort, *,
        store: ExperimentStore | None = None,
        config: OptimizationConfig | None = None,
        guard: PromptGuard | None = None,
    ) -> None:
        self.harness = harness
        self.generator = generator
        self.store = store or ExperimentStore()
        self.config = config or OptimizationConfig()
        self.guard = guard or PromptGuard()
        self.selector = SelectionPolicy(self.config.selection)
        self.analyzer = BehaviorAnalyzer()
        self.attribution = PromptAttribution()
        self.extractor = LessonExtractor()

    def optimize(
        self, initial_state: PromptState,
        optimize_tasks: Sequence[Mapping[str, Any]],
        validation_tasks: Sequence[Mapping[str, Any]],
    ) -> OptimizationSessionResult:
        if not optimize_tasks or not validation_tasks:
            raise ValueError("Separate, non-empty optimize and validation splits are required")
        optimize_ids = {str(row.get("id", index)) for index, row in enumerate(optimize_tasks, 1)}
        validation_ids = {str(row.get("id", index)) for index, row in enumerate(validation_tasks, 1)}
        if optimize_ids & validation_ids:
            raise ValueError("Optimize and validation task identifiers must not overlap")
        self.guard.renderer.validate_bindable(initial_state)
        folder = self.store.create_session()
        state = initial_state
        seen: set[str] = {state.prompt_version_id}
        used_calls = 0
        accepted = 0
        stop_reason = "max_iterations"
        baseline_validation_score = 0.0
        validation_report: EvaluationReport | None = None
        write_json(folder / "session.json", {
            "status": "running", "initial_version": initial_state.prompt_version_id,
            "seed": self.config.seed, "max_iterations": self.config.max_iterations,
            "candidates_per_iteration": self.config.candidates_per_iteration,
        })
        try:
            self.store.version(folder, state)
            optimize_report = self._evaluate(folder, state, optimize_tasks, "optimize", "baseline")
            validation_report = self._evaluate(folder, state, validation_tasks, "validation", "baseline")
            baseline_validation_score = validation_report.score
            for number in range(1, self.config.max_iterations + 1):
                patterns = self.analyzer.analyze(optimize_report)
                if not patterns:
                    stop_reason = "no_failures"
                    break
                if used_calls >= self.config.max_llm_calls:
                    stop_reason = "llm_budget_exhausted"
                    break
                step = folder / "iterations" / f"{number:03d}"
                step.mkdir()
                write_json(step / "diagnosis.json", [
                    {"pattern_id": p.pattern_id, "kind": p.kind, "tasks": list(p.affected_tasks),
                     "evidence_refs": list(p.evidence_refs), "impact": p.impact,
                     "attribution": vars(self.attribution.attribute(p, state))}
                    for p in patterns
                ])
                lessons = []
                for pattern in sorted(patterns, key=lambda p: -p.impact):
                    diagnosis = self.attribution.attribute(pattern, state)
                    lesson = self.extractor.extract(pattern, diagnosis)
                    if lesson is not None:
                        lessons.append(lesson)
                if not lessons:
                    stop_reason = "not_prompt_fixable"
                    break
                attempts: list[dict[str, Any]] = []
                viable: list[tuple[PromptState, EvaluationReport, EvaluationReport, float, str]] = []
                for index in range(self.config.candidates_per_iteration):
                    if used_calls >= self.config.max_llm_calls:
                        break
                    lesson = lessons[index % len(lessons)]
                    used_calls += 1
                    entry: dict[str, Any] = {"candidate_number": index + 1, "lesson": vars(lesson)}
                    attempts.append(entry)
                    try:
                        patch = self.generator.generate(state, lesson)
                        entry["patch"] = patch.to_dict()
                        candidate = self.guard.apply(state, patch)
                        entry["prompt_version_id"] = candidate.prompt_version_id
                        if candidate.prompt_version_id in seen:
                            raise ValueError("Duplicate prompt candidate")
                        seen.add(candidate.prompt_version_id)
                        self.store.version(folder, candidate)
                        run_label = f"iteration-{number:03d}-candidate-{index + 1:02d}"
                        candidate_optimize = self._evaluate(folder, candidate, optimize_tasks, "optimize", run_label)
                        candidate_validation = self._evaluate(folder, candidate, validation_tasks, "validation", run_label)
                        decision = self.selector.compare(validation_report, candidate_validation)
                        entry["selection"] = vars(decision)
                        entry["optimize_score"] = candidate_optimize.score
                        entry["validation_score"] = candidate_validation.score
                        if decision.accepted:
                            viable.append((candidate, candidate_optimize, candidate_validation, decision.delta, patch.patch_id))
                        entry["status"] = "eligible" if decision.accepted else "rejected"
                    except (ValueError, TypeError, RuntimeError, KeyError) as exc:
                        entry["status"] = "rejected"
                        entry["reason"] = f"{type(exc).__name__}: {exc}"
                    finally:
                        write_json(step / "candidates.json", attempts)
                if not viable:
                    stop_reason = "no_accepted_candidate"
                    break
                best = max(viable, key=lambda item: (item[3], item[2].score, -sum(len(c.template) for c in item[0].components)))
                state, optimize_report, validation_report, _, chosen_patch_id = best
                accepted += 1
                for entry in attempts:
                    if entry.get("patch", {}).get("patch_id") == chosen_patch_id:
                        entry["status"] = "accepted"
                    elif entry.get("status") == "eligible":
                        entry["status"] = "rejected"
                        entry["reason"] = "Valid improvement, but a stronger candidate was selected"
                write_json(step / "candidates.json", attempts)
                write_json(step / "selected.json", {"version_id": state.prompt_version_id, "patch_id": chosen_patch_id})
            result = OptimizationSessionResult(
                session_dir=folder, baseline_version_id=initial_state.prompt_version_id,
                selected_version_id=state.prompt_version_id, accepted_iterations=accepted,
                stop_reason=stop_reason, baseline_validation_score=baseline_validation_score,
                selected_validation_score=validation_report.score,
            )
            write_json(folder / "session.json", {"status": "completed", **{k: str(v) if isinstance(v, Path) else v for k, v in vars(result).items()},
                                                    "llm_calls": used_calls})
            return result
        except BaseException as exc:
            write_json(folder / "session.json", {
                "status": "failed", "initial_version": initial_state.prompt_version_id,
                "last_selected_version": state.prompt_version_id,
                "error": f"{type(exc).__name__}: {exc}", "llm_calls": used_calls,
            })
            raise

    def _evaluate(
        self, folder: Path, state: PromptState,
        tasks: Sequence[Mapping[str, Any]], split: str, label: str,
    ) -> EvaluationReport:
        report = self.harness.evaluate(
            state, tasks, seed=self.config.seed, split=split,
            runs_root=folder / "agent_runs",
        )
        write_json(folder / "reports" / label / f"{split}.json", report.to_dict())
        return report
