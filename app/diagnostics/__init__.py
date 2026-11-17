"""Public diagnostics interface."""

from .core import diagnose, diagnose_case
from .classifier import ChatClassifier, Classification, FailureClassifier, JevClassifier
from .normalizer import normalize_trace
from .quality import QualityThresholds
from .rules import check_rules
from .skim import skim_case
from .models import (
    Confidence, DiagnosisBatchResult, DiagnosisReport, DiagnosisStatus,
    EvidenceRef, FailureType, SuggestedScope, TraceEvent, TraceEventType,
    DiagnosticSignal, SignalLevel,
    BehaviorPattern, ClassifierAttempt, DiagnosisSummary, Hypothesis,
)

__all__ = [
    "diagnose", "diagnose_case", "Confidence", "DiagnosisBatchResult", "DiagnosisReport",
    "DiagnosisStatus", "EvidenceRef", "FailureType", "SuggestedScope",
    "normalize_trace", "TraceEvent", "TraceEventType",
    "check_rules", "skim_case", "DiagnosticSignal", "SignalLevel",
    "BehaviorPattern", "DiagnosisSummary",
    "Hypothesis", "QualityThresholds",
    "ChatClassifier", "Classification", "ClassifierAttempt", "FailureClassifier", "JevClassifier",
]
