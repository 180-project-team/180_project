"""Deterministic observations over runner outcomes and normalized events."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Mapping, Sequence

from .models import (
    Confidence, DiagnosticSignal, EvidenceRef, FailureType, SignalLevel,
    TraceEvent, TraceEventType,
)


def check_rules(
    result: Mapping[str, Any],
    events: Sequence[TraceEvent],
    *,
    answer_validator: Callable[[Any], bool] | None = None,
) -> list[DiagnosticSignal]:
    """Return signals without selecting a critical step or diagnosing cause.

    Events must refer to this result and come from a compatible normalizer.
    A validator returns False for a known output-contract violation; exceptions
    propagate as caller/validator errors. Without it, arbitrary nonempty answers
    are accepted. Missing answers are checked only for completed (ok) runs.
    Search queries are compared exactly. Calls, results, then search history are
    preferred as a single source; partial sources can therefore miss repeats.
    """
    signals: list[DiagnosticSignal] = []
    if result.get("status") == "error":
        refs = [EvidenceRef(["status"])]
        if "error" in result:
            refs.append(EvidenceRef(["error"]))
        signals.append(DiagnosticSignal(
            "execution_error", FailureType.UNKNOWN, SignalLevel.FAILURE, None,
            refs, "Runner reported an execution error; the responsible component is unknown.",
            Confidence.HIGH,
        ))

    for event in events:
        if event.event_type == TraceEventType.TOOL_RESULT and event.error_ref is not None:
            signals.append(DiagnosticSignal(
                "tool_error", FailureType.TOOL_EXECUTION_FAILURE, SignalLevel.FAILURE,
                event.step_index, [event.error_ref],
                "The tool result reports an error; argument validity, recovery and root cause are not established.",
                Confidence.MEDIUM,
            ))

    if result.get("status") == "ok":
        answer = result.get("answer")
        empty = (answer is None or (isinstance(answer, str) and not answer.strip())
                 or (isinstance(answer, (list, dict, tuple)) and not answer))
        if empty:
            # A missing field cannot be an evidence path; status proves completion.
            refs = [EvidenceRef(["answer"] if "answer" in result else ["status"])]
            signals.append(DiagnosticSignal(
                "missing_answer", FailureType.OUTPUT_FORMAT_ERROR, SignalLevel.FAILURE,
                None, refs, "Runner completed successfully but the final answer is missing or empty.",
                Confidence.HIGH,
            ))
        elif answer_validator is not None and not answer_validator(answer):
            signals.append(DiagnosticSignal(
                "invalid_answer", FailureType.OUTPUT_FORMAT_ERROR, SignalLevel.FAILURE,
                None, [EvidenceRef(["answer"])],
                "The final answer fails the caller-supplied output contract.", Confidence.HIGH,
            ))

    for kind in (TraceEventType.TOOL_CALL, TraceEventType.TOOL_RESULT,
                 TraceEventType.SEARCH_OBSERVATION):
        searches = [event for event in events
                    if event.event_type == kind and event.search_query
                    and event.search_query.strip()]
        if searches:
            break
    groups: dict[str, list[TraceEvent]] = defaultdict(list)
    for event in searches:
        groups[event.search_query].append(event)
    for repeated in groups.values():
        if len(repeated) > 1:
            signals.append(DiagnosticSignal(
                "repeated_search", FailureType.INEFFICIENT_LOOP, SignalLevel.WARNING,
                repeated[1].step_index, [event.payload_ref for event in repeated],
                "The same search query occurs repeatedly; inefficiency and impact on the answer are unproven.",
                Confidence.LOW,
            ))
    return signals
