"""Normalize Skim Search steps and link events to their source fields."""

from __future__ import annotations

from typing import Any, Mapping

from .models import EvidenceRef, TraceEvent, TraceEventType


_SEARCH_ACTIONS = {"bm25_search", "bm25q_search", "dense_search", "hybrid_search", "reranked_search"}
_VISIT_ACTIONS = {"visit", "visit_q", "visit_d", "visit_h", "visit_r"}
_MODEL_ACTIONS = {"answer", "submit", "fix", "fix_rejected", "stop", "budget", "none"}


def _action_name(step: Mapping[str, Any]) -> str:
    action = step.get("action")
    return action.strip().lower() if isinstance(action, str) else ""


def _step_type(step: Mapping[str, Any], action: str) -> TraceEventType:
    if not action:
        return TraceEventType.UNKNOWN
    observation = step.get("observation")
    if action in _MODEL_ACTIONS or (isinstance(observation, str) and observation.strip() == "(episode ended)"):
        return TraceEventType.MODEL_RESPONSE
    if (action in _SEARCH_ACTIONS or action in _VISIT_ACTIONS
            or isinstance(step.get("args"), Mapping) or isinstance(observation, str)):
        return TraceEventType.TOOL_CALL
    return TraceEventType.UNKNOWN


def _search_query(step: Mapping[str, Any], action: str) -> str | None:
    if action not in _SEARCH_ACTIONS:
        return None
    args = step.get("args")
    if isinstance(args, Mapping):
        for field in ("query", "q"):
            query = args.get(field)
            if isinstance(query, str) and query.strip():
                return query
    query = step.get("query")
    return query if isinstance(query, str) and query.strip() else None


def _is_error_observation(observation: Any) -> bool:
    return isinstance(observation, str) and observation.lstrip().startswith("ERROR")


def normalize_trace(result: Mapping[str, Any]) -> list[TraceEvent]:
    """Return observations whose paths resolve within the supplied runner row.

    Terminal steps preserve their source positions as model responses. The
    runner's answer and error fields are represented once when present.
    """
    events: list[TraceEvent] = []

    def add(
        path: list[str | int],
        kind: TraceEventType,
        step_index: int | None = None,
        component: str | None = None,
        call_id: str | None = None,
        query: str | None = None,
        error_ref: EvidenceRef | None = None,
    ) -> None:
        events.append(TraceEvent(
            event_id="/" + "/".join(str(part) for part in path),
            step_index=step_index,
            event_type=kind,
            component_id=component,
            payload_ref=EvidenceRef(path),
            tool_call_id=call_id,
            search_query=query,
            error_ref=error_ref,
        ))

    trace = result.get("trajectory")
    if isinstance(trace, list):
        for index, step in enumerate(trace):
            path: list[str | int] = ["trajectory", index]
            if not isinstance(step, Mapping):
                add(path, TraceEventType.UNKNOWN, index)
                continue
            action = _action_name(step)
            kind = _step_type(step, action)
            if kind != TraceEventType.TOOL_CALL:
                add(path, kind, index, action or None)
                continue

            call_id = f"skim-step-{index}"
            query = _search_query(step, action)
            add(path, kind, index, action, call_id, query)
            observation = step.get("observation")
            if isinstance(observation, str):
                observation_path = path + ["observation"]
                error = EvidenceRef(observation_path) if _is_error_observation(observation) else None
                add(observation_path, TraceEventType.TOOL_RESULT, index, action, call_id, query, error)
    elif trace is not None:
        add(["trajectory"], TraceEventType.UNKNOWN)

    for field, kind in (("answer", TraceEventType.FINAL_ANSWER),
                        ("error", TraceEventType.EXECUTION_ERROR)):
        if result.get(field) is not None:
            add([field], kind)
    return events
