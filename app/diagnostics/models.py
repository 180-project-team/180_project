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


class TraceEventType(str, Enum):
    MODEL_RESPONSE = "MODEL_RESPONSE"
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"
    SEARCH_OBSERVATION = "SEARCH_OBSERVATION"
    FINAL_ANSWER = "FINAL_ANSWER"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    UNKNOWN = "UNKNOWN"


class SignalLevel(str, Enum):
    FAILURE = "FAILURE"
    WARNING = "WARNING"


@dataclass
class EvidenceRef:
    """A path into the original result, using field names and list indices."""

    path: list[str | int]


@dataclass
class TraceEvent:
    """One observation; identifiers and references are local to a runner row.

    step_index is the zero-based trajectory position, not MADQA's iteration.
    Events outside trajectory have no known step. event_id is derived from
    the source path, not an identifier supplied by the agent.
    """

    event_id: str
    step_index: int | None
    event_type: TraceEventType
    component_id: str | None
    payload_ref: EvidenceRef
    tool_call_id: str | None = None
    search_query: str | None = None
    error_ref: EvidenceRef | None = None


@dataclass
class DiagnosticSignal:
    """An observed violation or a warning, not a root-cause conclusion."""

    rule_id: str
    failure_type: FailureType
    level: SignalLevel
    step_index: int | None
    evidence_refs: list[EvidenceRef]
    rationale: str
    confidence: Confidence
    component_id: str | None = None


@dataclass
class DiagnosisReport:
    """Observed failures and warnings; FAILURE_FOUND need not mean final failure.

    A recovered tool error remains an observation. critical_step is only a
    candidate location and may be absent even when a failure was observed.
    """

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
    signals: list[DiagnosticSignal] = field(default_factory=list)


@dataclass
class BehaviorPattern:
    """A rule observed in multiple reports of one run and version."""

    rule_id: str
    failure_type: FailureType
    level: SignalLevel
    run_id: str | None
    version_id: str | None
    report_indices: list[int]


@dataclass
class DiagnosisSummary:
    """Indices refer to batch reports; counts use one category per failed case.

    Warning-only cases and diagnostic errors are excluded from failure counts.
    Individual reports preserve source identity in mixed batches.
    """

    analyzed_cases: int = 0
    failed_cases: int = 0
    failure_counts: dict[FailureType, int] = field(default_factory=dict)
    report_indices: list[int] = field(default_factory=list)
    patterns: list[BehaviorPattern] = field(default_factory=list)
    run_id: str | None = None
    version_id: str | None = None


@dataclass
class DiagnosisBatchResult:
    reports: list[DiagnosisReport] = field(default_factory=list)
    summary: DiagnosisSummary = field(default_factory=DiagnosisSummary)
