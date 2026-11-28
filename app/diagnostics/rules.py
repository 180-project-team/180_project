"""Deterministic observations over runner outcomes and normalized events."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Mapping, Sequence

from .models import (
    Confidence, DiagnosticSignal, EvidenceRef, FailureType, SignalLevel,
    TraceEvent, TraceEventType,
)
from .normalizer import _action_name, _is_error_observation, _search_query, _step_type

_INVALID_SKIM_OBSERVATIONS = (
    "ERROR: your <tool_call> block was empty.",
    "ERROR: the JSON inside your <tool_call> did not parse",
    "ERROR: no tool call found.",
    "ERROR: no prior search",
    "ERROR: rank ",
    "ERROR: ambiguous doc ",
    "ERROR: no such doc ",
    "ERROR: unknown tool ",
    "ERROR: invalid tool call (",
)
_BUDGET_STOPS = ("max_steps", "max_turns", "ctx_budget")


def _later_tool_result(
    steps: Sequence[Any], index: int, step: Mapping[str, Any], action: str,
) -> tuple[int | None, bool]:
    """Find a later result from this tool, preferring an observed exact search retry."""
    args = step.get("args")
    query = _search_query(step, action)
    first_result = None
    for later_index in range(index + 1, len(steps)):
        later = steps[later_index]
        if not isinstance(later, Mapping) or _action_name(later) != action:
            continue
        observation = later.get("observation")
        if (not isinstance(observation, str) or not observation.strip()
                or _is_error_observation(observation) or _step_type(later, action) != TraceEventType.TOOL_CALL):
            continue
        if (query is not None and isinstance(args, Mapping) and args and later.get("args") == args
                and _search_query(later, action) == query):
            return later_index, True
        if first_result is None:
            first_result = later_index
    return first_result, False


def _skim_execution_signals(
    result: Mapping[str, Any], events: Sequence[TraceEvent],
) -> list[DiagnosticSignal]:
    trace = result.get("trajectory")
    steps = trace if isinstance(trace, list) else []
    signals = []
    if result.get("stopped") in _BUDGET_STOPS:
        budget = next((event for event in reversed(events)
                       if event.event_type == TraceEventType.MODEL_RESPONSE and event.component_id == "budget"), None)
        refs = [EvidenceRef(["stopped"])]
        if budget is not None:
            refs.append(budget.payload_ref)
        signals.append(DiagnosticSignal(
            "budget_stop", FailureType.PREMATURE_STOP, SignalLevel.FAILURE,
            budget.step_index if budget is not None else None, refs,
            "The agent reached its execution budget.", Confidence.HIGH,
        ))

    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            continue
        action = _action_name(step)
        observation = step.get("observation")
        if not _is_error_observation(observation):
            continue
        is_tool = _step_type(step, action) == TraceEventType.TOOL_CALL
        if not is_tool and action != "none":
            continue
        invalid = observation.lstrip().startswith(_INVALID_SKIM_OBSERVATIONS)
        later_index, exact_retry = _later_tool_result(steps, index, step, action) if is_tool else (None, False)
        refs = [EvidenceRef(["trajectory", index, "observation"])]
        rationale = "Skim recorded an error for this action."
        if later_index is not None:
            refs.append(EvidenceRef(["trajectory", later_index, "observation"]))
            if exact_retry:
                rationale += " A later search with the same query and arguments returned a result."
            else:
                rationale += " The same tool returned a result at a later step."
            rationale += " The effect of the error on the final answer is unknown."
        level = SignalLevel.WARNING if exact_retry or action == "none" and not invalid else SignalLevel.FAILURE
        category = (FailureType.INVALID_TOOL_CALL if invalid else
                    FailureType.OUTPUT_FORMAT_ERROR if action == "none" else
                    FailureType.TOOL_EXECUTION_FAILURE)
        signals.append(DiagnosticSignal(
            "invalid_tool_call" if invalid else "model_turn_error" if action == "none" else "tool_error",
            category, level, index, refs, rationale,
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

    signals.extend(_skim_execution_signals(result, events))
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
