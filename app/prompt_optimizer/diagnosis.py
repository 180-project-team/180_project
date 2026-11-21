"""Evidence-backed failure analysis, attribution, and transferable lessons."""

from __future__ import annotations

from dataclasses import dataclass

from .evaluation import EvaluationReport
from .models import PromptState


@dataclass(frozen=True)
class BehaviorPattern:
    pattern_id: str
    kind: str
    affected_tasks: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    impact: float


@dataclass(frozen=True)
class PromptDiagnosis:
    pattern_id: str
    component_id: str | None
    confidence: float
    rationale: str
    status: str


@dataclass(frozen=True)
class PromptLesson:
    pattern_id: str
    component_id: str
    instruction: str
    evidence_refs: tuple[str, ...]


class BehaviorAnalyzer:
    def analyze(self, report: EvaluationReport) -> list[BehaviorPattern]:
        grouped: dict[str, list] = {}
        for row in report.items:
            if row.score >= 1.0:
                continue
            if row.status != "ok":
                kind = "execution_error"
            else:
                events_text = " ".join(str(value).casefold() for event in row.trace_events for value in event.values())
                if "format_violation" in events_text or "invalid_format" in events_text:
                    kind = "format_violation"
                elif "without_verification" in events_text or "skipped_verification" in events_text:
                    kind = "insufficient_verification"
                elif "tool_error" in events_text or "tool_failure" in events_text:
                    kind = "tool_failure"
                elif "premature_final" in events_text or "early_stop" in events_text:
                    kind = "premature_final_answer"
                else:
                    kind = "incorrect_answer"
            grouped.setdefault(kind, []).append(row)
        return [
            BehaviorPattern(
                pattern_id=f"{kind}-{len(rows)}", kind=kind,
                affected_tasks=tuple(row.task_id for row in rows),
                evidence_refs=tuple(row.trace_ref for row in rows),
                impact=sum(1.0 - row.score for row in rows),
            )
            for kind, rows in grouped.items()
        ]


class PromptAttribution:
    def attribute(self, pattern: BehaviorPattern, state: PromptState) -> PromptDiagnosis:
        if pattern.kind in {"execution_error", "tool_failure"}:
            return PromptDiagnosis(pattern.pattern_id, None, 1.0,
                                   "Agent or tool execution failed; prompt cause is unconfirmed", "NOT_PROMPT_FIXABLE")
        by_id = {component.component_id: component for component in state.components if component.editable}
        preferred = {
            "format_violation": ("output.format", "system.core"),
            "premature_final_answer": ("planner.instructions", "system.core"),
            "insufficient_verification": ("verifier.instructions", "system.core"),
        }.get(pattern.kind, ("verifier.instructions", "system.core", "planner.instructions", "output.format"))
        for name in preferred:
            if name in by_id:
                return PromptDiagnosis(pattern.pattern_id, name, 0.5,
                                       "Generic incorrect answer: provisional prompt attribution", "PROMPT_FIXABLE")
        component = next((c for c in state.components if c.editable), None)
        if component is None:
            return PromptDiagnosis(pattern.pattern_id, None, 1.0, "No editable components", "NOT_PROMPT_FIXABLE")
        return PromptDiagnosis(pattern.pattern_id, component.component_id, 0.25,
                               "Fallback attribution; requires experimental validation", "PROMPT_FIXABLE")


class LessonExtractor:
    def extract(self, pattern: BehaviorPattern, diagnosis: PromptDiagnosis) -> PromptLesson | None:
        if diagnosis.status != "PROMPT_FIXABLE" or diagnosis.component_id is None:
            return None
        instructions = {
            "format_violation": "Check the required output format before returning the final answer.",
            "premature_final_answer": "Do not finish before completing the relevant evidence-gathering steps.",
            "insufficient_verification": "Before producing a final answer, verify that it follows available evidence and task requirements.",
            "incorrect_answer": "Before producing a final answer, verify that it follows available evidence and task requirements.",
        }
        return PromptLesson(
            pattern_id=pattern.pattern_id,
            component_id=diagnosis.component_id,
            instruction=instructions.get(pattern.kind, instructions["incorrect_answer"]),
            evidence_refs=pattern.evidence_refs,
        )
