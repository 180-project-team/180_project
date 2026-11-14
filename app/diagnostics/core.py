"""Entry point for diagnosing existing runner results."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .models import (
    Confidence, DiagnosisBatchResult, DiagnosisReport, DiagnosisStatus,
    FailureType, SuggestedScope,
)


def diagnose(
    results: Iterable[Mapping[str, Any]],
    evaluations: Any = None,
    *,
    run_id: str | None = None,
    version_id: str | None = None,
) -> DiagnosisBatchResult:
    """Create one placeholder report per result, preserving input order.

    Evaluations are reserved for a future integration and are not used yet.
    No diagnostic checks are performed at this stage. Each result must be a
    mapping; its optional ``id`` is expected to be a string, integer, or None.
    """
    return DiagnosisBatchResult(reports=[
        DiagnosisReport(
            case_id=row.get("id"),
            run_id=run_id,
            version_id=version_id,
            status=DiagnosisStatus.INSUFFICIENT_DATA,
            failure_type=FailureType.UNKNOWN,
            critical_step=None,
            evidence_refs=[],
            rationale="Diagnostic checks are not implemented yet.",
            confidence=Confidence.LOW,
            suggested_scope=SuggestedScope.UNKNOWN,
        )
        for row in results
    ])
