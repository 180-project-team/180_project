"""Coordinate normalization, deterministic checks and batch reporting."""

from __future__ import annotations

import json
import math
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
    primary = next(
        (signal for signal in failures if signal.failure_type != FailureType.UNKNOWN),
        failures[0] if failures else None,
    )
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
        if primary.rule_id == "budget_stop":
            critical_step = primary.step_index
        elif (row.get("status") == "error" and len(tool_errors) == 1
                and primary.rule_id == "tool_error"
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
        FailureType.INVALID_TOOL_CALL: SuggestedScope.PROMPT,
        FailureType.PREMATURE_STOP: SuggestedScope.STRUCTURAL,
        FailureType.TOOL_EXECUTION_FAILURE: SuggestedScope.STRUCTURAL,
    }.get(category, SuggestedScope.UNKNOWN)
    refs: list[EvidenceRef] = []
    for signal in signals:
        for ref in signal.evidence_refs:
            if ref not in refs:
                refs.append(ref)
    return DiagnosisReport(
        case_id=row.get("case_id", row.get("id")), run_id=run_id, version_id=version_id,
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
                case_id=row.get("case_id", row.get("id")), run_id=case_run, version_id=case_version,
                status=DiagnosisStatus.DIAGNOSTIC_ERROR, failure_type=FailureType.UNKNOWN,
                critical_step=None, evidence_refs=[],
                rationale=f"Diagnostics failed: {type(exc).__name__}: {exc}",
                confidence=Confidence.LOW, suggested_scope=SuggestedScope.UNKNOWN,
            )
        reports.append(report)
    return DiagnosisBatchResult(reports=reports, summary=summarize(reports))


def diagnose_case(case: str | Mapping[str, Any]) -> DiagnosisReport:
    """Validate one JSON case and report evidence relative to that object."""
    row = json.loads(case) if isinstance(case, str) else case
    if not isinstance(row, Mapping):
        raise ValueError("A diagnostic case must be a JSON object")
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if "answer" not in row:
        raise ValueError("answer is required and may be null")
    if row.get("status") not in ("ok", "error", "unknown"):
        raise ValueError("status must be ok, error, or unknown")
    if "trajectory" not in row or (
        row["trajectory"] is not None and not isinstance(row["trajectory"], list)
    ):
        raise ValueError("trajectory must be an array or null")
    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("metrics must be an object")
    for name, value in metrics.items():
        if not isinstance(name, str) or not name:
            raise ValueError("metric names must be non-empty strings")
        if value is not None and (
            type(value) not in (int, float)
            or isinstance(value, float) and not math.isfinite(value)
        ):
            raise ValueError(f"Metric {name!r} must be a finite number or null")
    case_id = row.get("case_id")
    if case_id is not None and type(case_id) not in (int, str):
        raise ValueError("case_id must be a string, integer, or null")
    for field in ("run_id", "version_id", "traceback"):
        if row.get(field) is not None and not isinstance(row[field], str):
            raise ValueError(f"{field} must be a string or null")
    return diagnose([row]).reports[0]
