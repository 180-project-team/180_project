"""Normalize runner rows without copying payloads or inferring missing events.

Skim Search steps carry an action, arguments, and a tool observation.
Historical runner rows may carry separate tool_calls and tool_results.
"""

from __future__ import annotations

from typing import Any, Mapping

from .models import EvidenceRef, TraceEvent, TraceEventType


def normalize_trace(result: Mapping[str, Any]) -> list[TraceEvent]:
    """Return observations whose paths resolve within the supplied runner row.

    Missing/None trajectory produces no trajectory events. A non-list trace
    is retained as UNKNOWN. Search history is supplementary: it may describe
    the same searches as trajectory, and list order is not a combined timeline.
    Answer/error events represent present non-None fields, including empties;
    classifying their contents is the responsibility of diagnostic rules.
    """
    events: list[TraceEvent] = []

    def add(
        path: list[str | int],
        kind: TraceEventType,
        step_index: int | None = None,
        component: Any = None,
        call_id: Any = None,
        query: Any = None,
        error_ref: EvidenceRef | None = None,
    ) -> None:
        events.append(TraceEvent(
            event_id="/" + "/".join(str(part) for part in path),
            step_index=step_index,
            event_type=kind,
            component_id=component if isinstance(component, str) else None,
            payload_ref=EvidenceRef(path),
            tool_call_id=call_id if isinstance(call_id, str) else None,
            search_query=query if isinstance(query, str) else None,
            error_ref=error_ref,
        ))

    trace = result.get("trajectory")
    if isinstance(trace, list):
        for index, step in enumerate(trace):
            path: list[str | int] = ["trajectory", index]
            if not isinstance(step, Mapping):
                add(path, TraceEventType.UNKNOWN, index)
                continue
            action = step.get("action")
            if isinstance(action, str):
                if action in {"answer", "submit"}:
                    add(path, TraceEventType.FINAL_ANSWER, index, action)
                elif action in {"none", "budget", "stop"}:
                    add(path, TraceEventType.MODEL_RESPONSE, index, action)
                else:
                    call_id = f"skim-step-{index}"
                    query = step.get("query") if action == "bm25_search" else None
                    add(path, TraceEventType.TOOL_CALL, index, action, call_id, query)
                    observation = step.get("observation")
                    if isinstance(observation, str):
                        observation_path = path + ["observation"]
                        error = (EvidenceRef(observation_path)
                                 if observation.startswith("ERROR") else None)
                        add(observation_path, TraceEventType.TOOL_RESULT, index,
                            action, call_id, query, error)
                continue
            # MADQA logs model responses, not the original model requests.
            kind = (TraceEventType.MODEL_RESPONSE
                    if step.get("role") == "assistant" else TraceEventType.UNKNOWN)
            add(path, kind, index)
            for field, event_type, id_field in (
                ("tool_calls", TraceEventType.TOOL_CALL, "id"),
                ("tool_results", TraceEventType.TOOL_RESULT, "tool_call_id"),
            ):
                entries = step.get(field)
                if isinstance(entries, list):
                    for number, entry in enumerate(entries):
                        entry_path = path + [field, number]
                        if isinstance(entry, Mapping):
                            arguments = entry.get("arguments")
                            query = (arguments.get("query") if isinstance(arguments, Mapping)
                                     and event_type == TraceEventType.TOOL_CALL
                                     else entry.get("query"))
                            error_ref = (EvidenceRef(entry_path + ["error"])
                                         if entry.get("error") else None)
                            add(entry_path, event_type, index,
                                entry.get("name"), entry.get(id_field),
                                query if entry.get("name") == "search_documents" else None,
                                error_ref)
                        else:
                            add(entry_path, TraceEventType.UNKNOWN, index)
                elif entries is not None:
                    add(path + [field], TraceEventType.UNKNOWN, index)
            if step.get("parsed_answer") is not None:
                add(path + ["parsed_answer"], TraceEventType.FINAL_ANSWER, index)
    elif trace is not None:
        add(["trajectory"], TraceEventType.UNKNOWN)

    history = result.get("search_history")
    if isinstance(history, list):
        for index, entry in enumerate(history):
            kind = (TraceEventType.SEARCH_OBSERVATION
                    if isinstance(entry, Mapping) and "query" in entry
                    else TraceEventType.UNKNOWN)
            add(["search_history", index], kind,
                query=entry.get("query") if isinstance(entry, Mapping) else None)
    elif history is not None:
        add(["search_history"], TraceEventType.UNKNOWN)

    for field, kind in (("answer", TraceEventType.FINAL_ANSWER),
                        ("error", TraceEventType.EXECUTION_ERROR)):
        if result.get(field) is not None:
            add([field], kind)
    return events
