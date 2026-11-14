"""Public diagnostics interface."""

from .core import diagnose
from .normalizer import normalize_trace
from .rules import check_rules
from .models import (
    Confidence, DiagnosisBatchResult, DiagnosisReport, DiagnosisStatus,
    EvidenceRef, FailureType, SuggestedScope, TraceEvent, TraceEventType,
    DiagnosticSignal, SignalLevel,
    BehaviorPattern, DiagnosisSummary,
)

__all__ = [
    "diagnose", "Confidence", "DiagnosisBatchResult", "DiagnosisReport",
    "DiagnosisStatus", "EvidenceRef", "FailureType", "SuggestedScope",
    "normalize_trace", "TraceEvent", "TraceEventType",
    "check_rules", "DiagnosticSignal", "SignalLevel",
    "BehaviorPattern", "DiagnosisSummary",
]
