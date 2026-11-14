"""Contract tests for the initial diagnostics interface."""

import copy
import json
import unittest
from dataclasses import asdict

from app.diagnostics import (
    Confidence, DiagnosisStatus, FailureType, SuggestedScope, TraceEventType,
    diagnose, normalize_trace,
)


class DiagnosticsContractTests(unittest.TestCase):
    def test_batch_preserves_identity_order_and_does_not_mutate_input(self):
        results = [
            {"id": "a", "status": "ok", "answer": "answer", "trajectory": [{"step": 1}]},
            {"id": 7, "status": "error", "error": "failure"},
            {},
        ]
        original = copy.deepcopy(results)
        batch = diagnose(results, evaluations={"unused": 1}, run_id="run", version_id="v1")

        self.assertEqual(results, original)
        self.assertEqual([report.case_id for report in batch.reports], ["a", 7, None])
        self.assertEqual([report.status for report in batch.reports], [
            DiagnosisStatus.NO_FAILURE_DETECTED, DiagnosisStatus.FAILURE_FOUND,
            DiagnosisStatus.INSUFFICIENT_DATA,
        ])
        for report in batch.reports:
            self.assertEqual((report.run_id, report.version_id), ("run", "v1"))
            self.assertEqual(report.failure_type, FailureType.UNKNOWN)
            self.assertEqual(report.suggested_scope, SuggestedScope.UNKNOWN)
            self.assertIsNone(report.critical_step)
        self.assertEqual(batch.summary.analyzed_cases, 3)
        self.assertEqual(batch.summary.failed_cases, 1)

    def test_empty_batch(self):
        self.assertEqual(diagnose([]).reports, [])

    def test_result_is_json_serializable(self):
        batch = diagnose([{"id": "case"}])
        decoded = json.loads(json.dumps(asdict(batch)))
        self.assertEqual(decoded["reports"][0]["status"], "INSUFFICIENT_DATA")
        self.assertIsNone(decoded["reports"][0]["run_id"])


class TraceNormalizerTests(unittest.TestCase):
    def test_madqa_events_preserve_paths_steps_and_tool_identity(self):
        result = {
            "id": "case",
            "trajectory": [
                {"iteration": 1, "role": "assistant", "content": None,
                 "tool_calls": [{"id": "call1", "name": "search_documents",
                                 "arguments": {"query": "revenue"}}],
                 "tool_results": [{"tool_call_id": "call1", "name": "search_documents",
                                   "query": "revenue", "results": []}]},
                {"iteration": 2, "role": "assistant",
                 "tool_calls": [{"id": "call2", "name": "provide_answer",
                                 "arguments": {"answer": ["42"], "citations": []}}],
                 "parsed_answer": {"answer": ["42"], "citations": []}},
            ],
            "search_history": [{"iteration": 1, "query": "revenue", "num_results": 0}],
            "answer": ["42"],
        }
        original = copy.deepcopy(result)
        events = normalize_trace(result)
        self.assertEqual([event.event_type for event in events], [
            TraceEventType.MODEL_RESPONSE, TraceEventType.TOOL_CALL,
            TraceEventType.TOOL_RESULT, TraceEventType.MODEL_RESPONSE,
            TraceEventType.TOOL_CALL, TraceEventType.FINAL_ANSWER,
            TraceEventType.SEARCH_OBSERVATION, TraceEventType.FINAL_ANSWER,
        ])
        self.assertEqual([event.step_index for event in events], [0, 0, 0, 1, 1, 1, None, None])
        self.assertEqual(events[1].component_id, "search_documents")
        self.assertEqual(events[1].tool_call_id, events[2].tool_call_id)
        self.assertEqual(events[1].tool_call_id, "call1")
        self.assertEqual(events[2].payload_ref.path, ["trajectory", 0, "tool_results", 0])
        self.assertEqual(len({event.event_id for event in events}), len(events))
        for event in events:
            value = result
            for part in event.payload_ref.path:
                value = value[part]
            self.assertIsNotNone(value)
        self.assertEqual(events, normalize_trace(result))
        self.assertEqual(result, original)
        json.dumps([asdict(event) for event in events])

    def test_missing_trace_does_not_invent_steps_or_errors(self):
        for result in ({}, {"trajectory": None}, {"trajectory": [], "status": "error"}):
            with self.subTest(result=result):
                self.assertEqual(normalize_trace(result), [])
        events = normalize_trace({"answer": "", "error": "TimeoutError"})
        self.assertEqual([event.payload_ref.path for event in events], [["answer"], ["error"]])
        self.assertTrue(all(event.step_index is None for event in events))

    def test_partial_and_unknown_entries_remain_addressable(self):
        result = {"trajectory": ["opaque", {}, {"role": "assistant", "tool_calls": [None, {}],
                   "tool_results": {"unexpected": "shape"}}],
                  "search_history": ["opaque", {"query": "test"}]}
        events = normalize_trace(result)
        self.assertEqual(events[0].event_type, TraceEventType.UNKNOWN)
        self.assertEqual(events[0].payload_ref.path, ["trajectory", 0])
        call = next(event for event in events if event.event_type == TraceEventType.TOOL_CALL)
        self.assertIsNone(call.component_id)
        self.assertIsNone(call.tool_call_id)
        self.assertTrue(all(event.step_index is None for event in events[-2:]))
        malformed = normalize_trace({"trajectory": "opaque", "search_history": {}})
        self.assertEqual([event.event_type for event in malformed], [TraceEventType.UNKNOWN] * 2)

    def test_tool_error_is_preserved_as_an_observation(self):
        result = {"trajectory": [{"role": "assistant", "tool_results": [
            {"tool_call_id": "call", "name": "search_documents", "error": "Invalid query format"}
        ]}]}
        event = normalize_trace(result)[1]
        self.assertEqual(event.event_type, TraceEventType.TOOL_RESULT)
        self.assertEqual(event.payload_ref.path, ["trajectory", 0, "tool_results", 0])
        self.assertEqual(event.step_index, 0)
        self.assertEqual(diagnose([result]).reports[0].status, DiagnosisStatus.FAILURE_FOUND)


if __name__ == "__main__":
    unittest.main()
