"""Normalize runner rows without copying payloads or inferring missing events.

Supported MADQA layout: trajectory entries with role/content, tool_calls
(id, name, arguments), tool_results (tool_call_id, name, results/error),
and parsed_answer. Source: OxRML/MADQA,
baselines/bm25-mllm-agent/vllm_agent.py. Unknown entries remain addressable.
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
    ) -> None:
        events.append(TraceEvent(
            event_id="/" + "/".join(str(part) for part in path),
            step_index=step_index,
            event_type=kind,
            component_id=component if isinstance(component, str) else None,
            payload_ref=EvidenceRef(path),
            tool_call_id=call_id if isinstance(call_id, str) else None,
        ))

    trace = result.get("trajectory")
    if isinstance(trace, list):
        for index, step in enumerate(trace):
            path: list[str | int] = ["trajectory", index]
            if not isinstance(step, Mapping):
                add(path, TraceEventType.UNKNOWN, index)
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
                            add(entry_path, event_type, index,
                                entry.get("name"), entry.get(id_field))
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
            add(["search_history", index], kind)
    elif history is not None:
        add(["search_history"], TraceEventType.UNKNOWN)

    for field, kind in (("answer", TraceEventType.FINAL_ANSWER),
                        ("error", TraceEventType.EXECUTION_ERROR)):
        if result.get(field) is not None:
            add([field], kind)
    return events
