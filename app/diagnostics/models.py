"""Public data contracts for the diagnostics module."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DiagnosisStatus(str, Enum):
    FAILURE_FOUND = "FAILURE_FOUND"
    NO_FAILURE_DETECTED = "NO_FAILURE_DETECTED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    DIAGNOSTIC_ERROR = "DIAGNOSTIC_ERROR"


class FailureType(str, Enum):
    INVALID_TOOL_CALL = "INVALID_TOOL_CALL"
    TOOL_EXECUTION_FAILURE = "TOOL_EXECUTION_FAILURE"
    RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
    CONTEXT_LOSS = "CONTEXT_LOSS"
    TOOL_OUTPUT_MISINTERPRETATION = "TOOL_OUTPUT_MISINTERPRETATION"
    INCOMPLETE_ANSWER = "INCOMPLETE_ANSWER"
    UNSUPPORTED_ANSWER = "UNSUPPORTED_ANSWER"
    CITATION_ERROR = "CITATION_ERROR"
    PREMATURE_STOP = "PREMATURE_STOP"
    INEFFICIENT_LOOP = "INEFFICIENT_LOOP"
    OUTPUT_FORMAT_ERROR = "OUTPUT_FORMAT_ERROR"
    UNKNOWN = "UNKNOWN"


class SuggestedScope(str, Enum):
    PROMPT = "PROMPT"
    STRUCTURAL = "STRUCTURAL"
    CODE = "CODE"
    UNKNOWN = "UNKNOWN"


class Confidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class EvidenceRef:
    """A path into the original result, using field names and list indices."""

    path: list[str | int]


@dataclass
class DiagnosisReport:
    case_id: str | int | None
    run_id: str | None
    version_id: str | None
    status: DiagnosisStatus
    failure_type: FailureType
    critical_step: int | None
    evidence_refs: list[EvidenceRef]
    rationale: str
    confidence: Confidence
    suggested_scope: SuggestedScope


@dataclass
class DiagnosisBatchResult:
    reports: list[DiagnosisReport] = field(default_factory=list)
