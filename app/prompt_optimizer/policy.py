"""Prompt guards and optional comparison of externally computed scores."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.prompt_optimizer.evaluation_report import EvaluationReport
from app.prompt_optimizer.models import PromptComponent, PromptPatch, PromptState, digest
from app.prompt_optimizer.renderer import PromptRenderer, placeholders

_RULE_RE = re.compile(r"^\s*- \[([A-Za-z0-9_.-]+)\] (.*)$")
_SECRET_RE = re.compile(r"(?:github_pat_[A-Za-z0-9_]{10,}|gh[pousr]_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,})")


class PromptGuard:
    """Never alter fixed components or silently drop template placeholders."""

    def __init__(self, max_growth_chars: int = 1500, renderer: PromptRenderer | None = None) -> None:
        if max_growth_chars < 0:
            raise ValueError("max_growth_chars must be non-negative")
        self.max_growth_chars = max_growth_chars
        self.renderer = renderer or PromptRenderer()

    def apply(self, state: PromptState, patch: PromptPatch) -> PromptState:
        component = state.get(patch.component_id)
        if not component.editable:
            raise ValueError(f"Fixed prompt component: {component.component_id}")
        if digest(component.template) != patch.before_hash:
            raise ValueError("Stale patch: before_hash does not match current component")
        if not patch.rationale.strip() or not patch.evidence_refs:
            raise ValueError("Patch requires rationale and evidence references")
        old = component.template
        if patch.operation == "REPLACE_COMPONENT":
            changed = patch.payload
        elif patch.operation == "APPEND_RULE":
            if "\n" in patch.payload or not patch.payload.strip():
                raise ValueError("APPEND_RULE requires one non-empty line")
            if not patch.rule_id or not re.fullmatch(r"[A-Za-z0-9_.-]+", patch.rule_id):
                raise ValueError("APPEND_RULE needs a safe rule_id")
            if any(rule_id == patch.rule_id for rule_id, _ in self._rules(old)):
                raise ValueError("Duplicate rule_id")
            changed = old.rstrip() + f"\n- [{patch.rule_id}] {patch.payload.strip()}\n"
        elif patch.operation in {"UPDATE_RULE", "REMOVE_RULE"}:
            if not patch.rule_id:
                raise ValueError("Rule edit requires rule_id")
            if patch.operation == "UPDATE_RULE" and ("\n" in patch.payload or not patch.payload.strip()):
                raise ValueError("UPDATE_RULE requires one non-empty line")
            lines = old.splitlines()
            matches = [i for i, line in enumerate(lines) if (match := _RULE_RE.match(line)) and match.group(1) == patch.rule_id]
            if len(matches) != 1:
                raise ValueError("Rule edit requires exactly one matching rule")
            index = matches[0]
            if patch.operation == "REMOVE_RULE":
                del lines[index]
            else:
                lines[index] = f"- [{patch.rule_id}] {patch.payload.strip()}"
            changed = "\n".join(lines) + ("\n" if old.endswith("\n") else "")
        else:
            raise ValueError(f"Unsupported patch operation: {patch.operation}")
        if not changed.strip() or changed == old:
            raise ValueError("Patch must make a non-empty change")
        if len(changed) > component.max_length or len(changed) - len(old) > self.max_growth_chars:
            raise ValueError("Prompt length or growth limit exceeded")
        if _SECRET_RE.search(changed):
            raise ValueError("Potential secret embedded in candidate prompt")
        if placeholders(changed) != placeholders(old):
            raise ValueError("Patch cannot change declared template placeholders")
        updated = PromptComponent(
            component_id=component.component_id, role=component.role, template=changed,
            variables=component.variables, editable=component.editable, max_length=component.max_length,
        )
        candidate = PromptState(
            components=tuple(updated if c.component_id == component.component_id else c for c in state.components),
            parent_id=state.prompt_version_id, renderer_version=state.renderer_version,
            metadata=state.metadata,
        )
        self.renderer.validate_bindable(candidate)
        return candidate

    @staticmethod
    def _rules(template: str) -> list[tuple[str, str]]:
        return [(match.group(1), match.group(2)) for line in template.splitlines() if (match := _RULE_RE.match(line))]


@dataclass(frozen=True)
class SelectionConfig:
    min_delta: float = 0.01
    max_regressions: int = 0
    max_token_cost_ratio: float | None = 1.15

    def __post_init__(self) -> None:
        if self.min_delta < 0 or self.max_regressions < 0:
            raise ValueError("Selection thresholds cannot be negative")
        if self.max_token_cost_ratio is not None and self.max_token_cost_ratio <= 0:
            raise ValueError("max_token_cost_ratio must be positive")


@dataclass(frozen=True)
class SelectionDecision:
    accepted: bool
    reason: str
    delta: float | None
    regressions: int | None
    status: str = "compared"


class SelectionPolicy:
    def __init__(self, config: SelectionConfig | None = None) -> None:
        self.config = config or SelectionConfig()

    def compare(self, parent: "EvaluationReport", candidate: "EvaluationReport") -> SelectionDecision:
        """Compare complete saved measurements; do not infer missing scores."""

        if not isinstance(parent, EvaluationReport) or not isinstance(candidate, EvaluationReport):
            raise TypeError("SelectionPolicy requires evaluation reports")
        before = {row.task_id: row.score for row in parent.items}
        after = {row.task_id: row.score for row in candidate.items}
        if len(before) != len(parent.items) or len(after) != len(candidate.items):
            raise ValueError("Evaluation reports must not contain duplicate task ids")
        if parent.split != candidate.split:
            raise ValueError("Parent and candidate must use the same evaluation split")
        if not before or before.keys() != after.keys():
            raise ValueError("Parent and candidate must be evaluated on identical, non-empty tasks")
        missing = sorted({row.task_id for report in (parent, candidate) for row in report.items
                          if row.score is None or row.status != "ok"})
        if missing:
            return SelectionDecision(
                False, "Insufficient evaluation data for tasks: " + ", ".join(missing),
                None, None, status="insufficient_data",
            )
        regressions = sum(after[task_id] < value - 1e-9 for task_id, value in before.items())
        delta = candidate.score - parent.score
        if delta < self.config.min_delta - 1e-9:
            return SelectionDecision(False, "Primary score improvement is insufficient", delta, regressions)
        if regressions > self.config.max_regressions:
            return SelectionDecision(False, "Too many regressed tasks", delta, regressions)
        if self.config.max_token_cost_ratio is not None and parent.total_tokens is not None and candidate.total_tokens is not None:
            if parent.total_tokens == 0:
                if candidate.total_tokens > 0:
                    return SelectionDecision(False, "Token cost increased from zero", delta, regressions)
            elif candidate.total_tokens / parent.total_tokens > self.config.max_token_cost_ratio:
                return SelectionDecision(False, "Token cost threshold exceeded", delta, regressions)
        return SelectionDecision(True, "Validated improvement", delta, regressions)
