"""Public diagnostics interface."""

from .core import diagnose, diagnose_case
from .artifacts import diagnose_run, load_diagnostic_cases
from .classifier import Classification, FailureClassifier, LLMClassifier
from .normalizer import normalize_trace
from .quality import QualityThresholds
from .rules import check_rules
from .models import (
    Confidence, DiagnosisBatchResult, DiagnosisReport, DiagnosisStatus,
    EvidenceRef, FailureType, SuggestedScope, TraceEvent, TraceEventType,
    DiagnosticSignal, SignalLevel,
    BehaviorPattern, ClassifierAttempt, DiagnosisSummary, Hypothesis,
)

__all__ = [
    "diagnose", "diagnose_case", "diagnose_run", "load_diagnostic_cases",
    "Confidence", "DiagnosisBatchResult", "DiagnosisReport",
    "DiagnosisStatus", "EvidenceRef", "FailureType", "SuggestedScope",
    "normalize_trace", "TraceEvent", "TraceEventType",
    "check_rules", "DiagnosticSignal", "SignalLevel",
    "BehaviorPattern", "DiagnosisSummary",
    "Hypothesis", "QualityThresholds",
    "Classification", "ClassifierAttempt", "FailureClassifier", "LLMClassifier",
]
