"""Generate a prompt candidate from the shared diagnostics report."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from app.prompt_optimizer.diagnostic_report import prompt_hypotheses
from app.prompt_optimizer.generator import PromptModelPort
from app.prompt_optimizer.models import PromptLesson, PromptPatch, PromptState
from app.prompt_optimizer.policy import PromptGuard

if TYPE_CHECKING:
    from app.diagnostics.models import DiagnosisReport


def propose_candidate(
    state: PromptState,
    diagnosis: DiagnosisReport | Mapping[str, Any],
    generator: PromptModelPort,
    *,
    guard: PromptGuard | None = None,
    primary_scope_only: bool = False,
) -> tuple[PromptState, PromptPatch] | None:
    """Apply one recommendation from the shared diagnostics report."""
    report = diagnosis if isinstance(diagnosis, Mapping) else asdict(diagnosis)
    hypothesis = next(iter(prompt_hypotheses(report, primary_scope_only=primary_scope_only)), None)
    editable_ids = [component.component_id for component in state.components if component.editable]
    if hypothesis is None or not editable_ids or not report["evidence_refs"]:
        return None

    matching_signals = (
        signal for signal in report.get("signals", ())
        if signal.get("failure_type") == report["failure_type"]
    )
    component_id = next(
        (signal["component_id"] for signal in matching_signals if signal.get("component_id") in editable_ids),
        None,
    )
    if component_id is None:
        component_id = "system.core" if "system.core" in editable_ids else editable_ids[0]

    run_ref = f"{report.get('run_id') or 'run'}#{report.get('case_id')}"
    evidence_refs = tuple(
        f"{run_ref}:{'.'.join(map(str, ref['path']))}" for ref in report["evidence_refs"]
    )
    lesson = PromptLesson(
        pattern_id=hypothesis["hypothesis_id"],
        component_id=component_id,
        instruction=hypothesis["advice"],
        evidence_refs=evidence_refs,
    )
    prompt_guard = guard or PromptGuard()
    prompt_guard.renderer.validate_bindable(state)
    patch = generator.generate(state, lesson)
    return prompt_guard.apply(state, patch), patch

