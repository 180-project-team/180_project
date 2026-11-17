"""Contract tests for the initial diagnostics interface."""

import copy
import json
import unittest
from dataclasses import asdict

from app.diagnostics import (
    Confidence, DiagnosisStatus, FailureType, SuggestedScope, TraceEventType,
    diagnose, diagnose_case, normalize_trace,
)


class DiagnosticsContractTests(unittest.TestCase):
    def test_json_case_matches_batch_and_preserves_evidence_paths(self):
        row = {
            "case_id": "q1", "question": "When?", "answer": None,
            "status": "error", "trajectory": None, "metrics": {"answer_em": None},
            "error": "timeout",
        }
        report = diagnose_case(json.dumps(row))
        self.assertEqual(report, diagnose([row]).reports[0])
        self.assertEqual(report.case_id, "q1")
        self.assertIn(["error"], [ref.path for ref in report.evidence_refs])

    def test_json_case_rejects_invalid_metric_values(self):
        row = {
            "question": "When?", "answer": "Today",
            "status": "ok", "trajectory": [], "metrics": {},
        }
        for value in (True, "0.5", float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    diagnose_case(dict(row, metrics={"answer_f1": value}))

    def test_batch_preserves_identity_order_and_does_not_mutate_input(self):
        results = [
            {"id": "a", "status": "ok", "answer": "answer", "trajectory": [{"step": 1}]},
            {"id": 7, "status": "error", "error": "failure"},
            {},
        ]
        original = copy.deepcopy(results)
        batch = diagnose(results, run_id="run", version_id="v1")

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
    def test_skim_events_preserve_paths_steps_and_tool_identity(self):
        result = {
            "trajectory": [
                {"action": "bm25_search", "args": {"query": "revenue"}, "query": "revenue",
                 "observation": "One result", "hit_ids": ["report"]},
                {"action": "visit", "args": {"rank": 1}, "observation": "Revenue was 42.",
                 "read_ids": ["report"]},
                {"action": "answer"},
            ],
            "answer": "42",
        }
        original = copy.deepcopy(result)
        events = normalize_trace(result)
        self.assertEqual([event.event_type for event in events], [
            TraceEventType.TOOL_CALL, TraceEventType.TOOL_RESULT,
            TraceEventType.TOOL_CALL, TraceEventType.TOOL_RESULT,
            TraceEventType.FINAL_ANSWER, TraceEventType.FINAL_ANSWER,
        ])
        self.assertEqual([event.step_index for event in events], [0, 0, 1, 1, 2, None])
        self.assertEqual(events[0].component_id, "bm25_search")
        self.assertEqual(events[0].tool_call_id, events[1].tool_call_id)
        self.assertEqual(events[1].payload_ref.path, ["trajectory", 0, "observation"])
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
        result = {"trajectory": ["opaque", {}, {"action": "visit"}]}
        events = normalize_trace(result)
        self.assertEqual([event.event_type for event in events], [
            TraceEventType.UNKNOWN, TraceEventType.UNKNOWN, TraceEventType.TOOL_CALL,
        ])
        self.assertEqual(events[0].payload_ref.path, ["trajectory", 0])
        self.assertEqual(events[-1].component_id, "visit")
        event, = normalize_trace({"trajectory": "opaque"})
        self.assertEqual(event.event_type, TraceEventType.UNKNOWN)

    def test_tool_error_is_preserved_as_an_observation(self):
        result = {"trajectory": [{"action": "bm25_search", "observation": "ERROR: timeout"}]}
        _, event = normalize_trace(result)
        self.assertEqual(event.event_type, TraceEventType.TOOL_RESULT)
        self.assertEqual(event.error_ref.path, ["trajectory", 0, "observation"])
        self.assertEqual(event.step_index, 0)
        self.assertEqual(diagnose([result]).reports[0].status, DiagnosisStatus.FAILURE_FOUND)


if __name__ == "__main__":
    unittest.main()
