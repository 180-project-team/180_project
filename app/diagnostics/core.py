"""Coordinate normalization, deterministic checks and batch reporting."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Sequence

from .aggregator import summarize
from .models import (
    Confidence, DiagnosisBatchResult, DiagnosisReport, DiagnosisStatus,
    DiagnosticSignal, EvidenceRef, FailureType, SignalLevel, SuggestedScope,
    TraceEvent, TraceEventType,
)
from .normalizer import normalize_trace
from .rules import check_rules


def _report(
    row: Mapping[str, Any], events: Sequence[TraceEvent],
    signals: list[DiagnosticSignal], run_id: str | None, version_id: str | None,
) -> DiagnosisReport:
    failures = [signal for signal in signals if signal.level == SignalLevel.FAILURE]
    primary = next((signal for signal in failures if signal.rule_id == "tool_error"),
                   failures[0] if failures else None)
    critical_step = None
    if primary is not None:
        status = DiagnosisStatus.FAILURE_FOUND
        category, confidence = primary.failure_type, primary.confidence
        rationale = primary.rationale
        # Only offer a terminal tool-error location in an explicitly failed run.
        # Later observations make recovery/causal attribution ambiguous.
        last_step = max((event.step_index for event in events
                         if event.step_index is not None), default=None)
        tool_errors = [signal for signal in failures if signal.rule_id == "tool_error"]
        if (row.get("status") == "error" and len(tool_errors) == 1
                and primary.step_index is not None and primary.step_index == last_step
                and not any(event.step_index == last_step and
                            (event.event_type == TraceEventType.FINAL_ANSWER or
                             (event.event_type == TraceEventType.TOOL_RESULT and event.error_ref is None))
                            for event in events)):
            critical_step = primary.step_index
            rationale += " This terminal tool-error step is a candidate failure location, not a proven root cause."
        else:
            rationale += " A critical step cannot be established from the available observations."
    elif row.get("status") == "ok":
        status = DiagnosisStatus.NO_FAILURE_DETECTED
        category, confidence = FailureType.UNKNOWN, Confidence.LOW
        rationale = "No failure detected by the available checks; answer correctness is not established."
    else:
        status = DiagnosisStatus.INSUFFICIENT_DATA
        category, confidence = FailureType.UNKNOWN, Confidence.LOW
        rationale = "No failure established; completion status or final-answer data is insufficient."

    # Review hints, not automatic optimization decisions.
    scope = {
        FailureType.OUTPUT_FORMAT_ERROR: SuggestedScope.PROMPT,
        FailureType.TOOL_EXECUTION_FAILURE: SuggestedScope.STRUCTURAL,
    }.get(category, SuggestedScope.UNKNOWN)
    refs: list[EvidenceRef] = []
    for signal in signals:
        for ref in signal.evidence_refs:
            if ref not in refs:
                refs.append(ref)
    return DiagnosisReport(
        case_id=row.get("id"), run_id=run_id, version_id=version_id,
        status=status, failure_type=category, critical_step=critical_step,
        evidence_refs=refs, rationale=rationale, confidence=confidence,
        suggested_scope=scope, signals=signals,
    )


def diagnose(
    results: Iterable[Mapping[str, Any]],
    evaluations: Any = None,
    *,
    run_id: str | None = None,
    version_id: str | None = None,
    normalizer: Callable[[Mapping[str, Any]], Sequence[TraceEvent]] = normalize_trace,
    answer_validator: Callable[[Any], bool] | None = None,
) -> DiagnosisBatchResult:
    """Diagnose runner rows in order without mutating them.

    Batch identifiers override per-row identifiers when supplied. Evaluations
    remain reserved and unused; low metrics alone do not establish a cause.
    Normalizer/validator exceptions become DIAGNOSTIC_ERROR for that case;
    errors iterating the batch and non-mapping rows propagate to the caller.
    """
    reports: list[DiagnosisReport] = []
    for row in results:
        if not isinstance(row, Mapping):
            raise TypeError("Each diagnostic result must be a mapping")
        case_run = run_id if run_id is not None else row.get("run_id")
        case_version = version_id if version_id is not None else row.get("version_id")
        try:
            events = list(normalizer(row))
            signals = check_rules(row, events, answer_validator=answer_validator)
            report = _report(row, events, signals, case_run, case_version)
        except Exception as exc:
            report = DiagnosisReport(
                case_id=row.get("id"), run_id=case_run, version_id=case_version,
                status=DiagnosisStatus.DIAGNOSTIC_ERROR, failure_type=FailureType.UNKNOWN,
                critical_step=None, evidence_refs=[],
                rationale=f"Diagnostics failed: {type(exc).__name__}: {exc}",
                confidence=Confidence.LOW, suggested_scope=SuggestedScope.UNKNOWN,
            )
        reports.append(report)
    return DiagnosisBatchResult(reports=reports, summary=summarize(reports))
