"""Deterministic observations over runner outcomes and normalized events."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Mapping, Sequence

from .models import (
    Confidence, DiagnosticSignal, EvidenceRef, FailureType, SignalLevel,
    TraceEvent, TraceEventType,
)

_INVALID_SKIM_OBSERVATIONS = (
    "ERROR: your <tool_call> block was empty.",
    "ERROR: the JSON inside your <tool_call> did not parse",
    "ERROR: no tool call found.",
    "ERROR: no prior search",
    "ERROR: rank ",
    "ERROR: ambiguous doc ",
    "ERROR: no such doc ",
)
_BUDGET_STOPS = ("max_steps", "max_turns", "ctx_budget")


def _skim_execution_signals(result: Mapping[str, Any]) -> list[DiagnosticSignal]:
    trace = result.get("trajectory")
    steps = trace if isinstance(trace, list) else []
    signals = []
    if result.get("trace_format") == "skim_search" and result.get("stopped") in _BUDGET_STOPS:
        signals.append(DiagnosticSignal(
            "budget_stop", FailureType.PREMATURE_STOP, SignalLevel.FAILURE,
            len(steps) - 1 if steps else None, [EvidenceRef(["stopped"])],
            "The agent reached its execution budget.", Confidence.HIGH,
        ))

    for index, step in enumerate(steps):
        if not isinstance(step, Mapping) or not isinstance(step.get("action"), str):
            continue
        action = step["action"]
        observation = step.get("observation")
        if not isinstance(observation, str) or not observation.startswith("ERROR"):
            continue
        if action in {"budget", "answer", "submit", "stop"}:
            continue
        invalid = observation.startswith(_INVALID_SKIM_OBSERVATIONS)
        # The same visit rank may resolve to a different document after a new search.
        args = step.get("args")
        retry = next((
            later_index for later_index, later in enumerate(steps[index + 1:], index + 1)
            if action == "bm25_search" and isinstance(args, Mapping) and args
            and isinstance(later, Mapping) and later.get("action") == action
            and later.get("args") == args and isinstance(later.get("observation"), str)
            and later["observation"] and not later["observation"].startswith("ERROR")
        ), None)
        refs = [EvidenceRef(["trajectory", index, "observation"])]
        if retry is not None:
            refs.append(EvidenceRef(["trajectory", retry, "observation"]))
        level = SignalLevel.WARNING if retry is not None or action == "none" and not invalid else SignalLevel.FAILURE
        category = (FailureType.INVALID_TOOL_CALL if invalid else
                    FailureType.OUTPUT_FORMAT_ERROR if action == "none" else
                    FailureType.TOOL_EXECUTION_FAILURE)
        signals.append(DiagnosticSignal(
            "invalid_tool_call" if invalid else "model_turn_error" if action == "none" else "tool_error",
            category, level, index, refs,
            "A later attempt with the same search arguments completed."
            if retry is not None else "Skim recorded an error for this action.",
            Confidence.HIGH if invalid else Confidence.MEDIUM, action,
        ))
    return signals


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
    Search queries are compared exactly. Results are used when an event stream
    contains no search calls.
    """
    signals: list[DiagnosticSignal] = []
    if result.get("status") == "error":
        refs = [EvidenceRef(["status"])]
        refs.extend(EvidenceRef([field]) for field in ("error", "traceback") if result.get(field))
        signals.append(DiagnosticSignal(
            "execution_error", FailureType.UNKNOWN, SignalLevel.FAILURE, None,
            refs, "Runner reported an execution error; the responsible component is unknown.",
            Confidence.HIGH,
        ))

    signals.extend(_skim_execution_signals(result))
    skim_refs = [ref for signal in signals for ref in signal.evidence_refs]
    for event in events:
        if (event.event_type == TraceEventType.TOOL_RESULT and event.error_ref is not None
                and event.error_ref not in skim_refs):
            signals.append(DiagnosticSignal(
                "tool_error", FailureType.TOOL_EXECUTION_FAILURE, SignalLevel.FAILURE,
                event.step_index, [event.error_ref],
                "The tool result reports an error; argument validity, recovery and root cause are not established.",
                Confidence.MEDIUM, event.component_id,
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

    for kind in (TraceEventType.TOOL_CALL, TraceEventType.TOOL_RESULT):
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
