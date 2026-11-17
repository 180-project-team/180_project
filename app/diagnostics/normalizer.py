"""Normalize Skim Search steps and link events to their source fields."""

from __future__ import annotations

from typing import Any, Mapping

from .models import EvidenceRef, TraceEvent, TraceEventType


def normalize_trace(result: Mapping[str, Any]) -> list[TraceEvent]:
    """Return observations whose paths resolve within the supplied runner row.

    Unknown steps retain their source positions. Answer and error events
    represent fields with non-None values.
    """
    events: list[TraceEvent] = []

    def add(
        path: list[str | int],
        kind: TraceEventType,
        step_index: int | None = None,
        component: str | None = None,
        call_id: str | None = None,
        query: Any = None,
        error_ref: EvidenceRef | None = None,
    ) -> None:
        events.append(TraceEvent(
            event_id="/" + "/".join(str(part) for part in path),
            step_index=step_index,
            event_type=kind,
            component_id=component,
            payload_ref=EvidenceRef(path),
            tool_call_id=call_id,
            search_query=query if isinstance(query, str) else None,
            error_ref=error_ref,
        ))

    trace = result.get("trajectory")
    if isinstance(trace, list):
        for index, step in enumerate(trace):
            path: list[str | int] = ["trajectory", index]
            action = step.get("action") if isinstance(step, Mapping) else None
            if not isinstance(action, str) or not action:
                add(path, TraceEventType.UNKNOWN, index)
            elif action in {"answer", "submit"}:
                add(path, TraceEventType.FINAL_ANSWER, index, action)
            elif action in {"none", "budget", "stop"}:
                add(path, TraceEventType.MODEL_RESPONSE, index, action)
            else:
                call_id = f"skim-step-{index}"
                query = step.get("query") if action == "bm25_search" else None
                add(path, TraceEventType.TOOL_CALL, index, action, call_id, query)
                if isinstance(observation := step.get("observation"), str):
                    observation_path = path + ["observation"]
                    error = EvidenceRef(observation_path) if observation.startswith("ERROR") else None
                    add(observation_path, TraceEventType.TOOL_RESULT, index, action, call_id, query, error)
    elif trace is not None:
        add(["trajectory"], TraceEventType.UNKNOWN)

    for field, kind in (("answer", TraceEventType.FINAL_ANSWER),
                        ("error", TraceEventType.EXECUTION_ERROR)):
        if result.get(field) is not None:
            add([field], kind)
    return events
