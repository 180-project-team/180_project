"""Contract tests for the initial diagnostics interface."""

import copy
import json
import unittest
from dataclasses import asdict
from pathlib import Path

from app.diagnostics import (
    DiagnosisStatus, FailureType, SuggestedScope, TraceEventType,
    diagnose, diagnose_case, normalize_trace,
)


class DiagnosticsContractTests(unittest.TestCase):
    def test_json_case_matches_batch_and_preserves_evidence_paths(self):
        row = {
            "case_id": "q1", "question": "When?", "answer": None,
            "status": "error", "trajectory": None, "metrics": {"fact_recall": None},
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
        for value in (True, "0.5", float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    diagnose_case(dict(row, metrics={"fact_recall": value}))

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
            TraceEventType.MODEL_RESPONSE, TraceEventType.FINAL_ANSWER,
        ])
        self.assertEqual([event.step_index for event in events], [0, 0, 1, 1, 2, None])
        self.assertEqual(events[0].component_id, "bm25_search")
        self.assertEqual(events[0].tool_call_id, events[1].tool_call_id)
        self.assertEqual(events[1].payload_ref.path, ["trajectory", 0, "observation"])
        self.assertEqual(events[-2].payload_ref.path, ["trajectory", 2])
        self.assertEqual(events[-1].payload_ref.path, ["answer"])
        self.assertEqual(len({event.event_id for event in events}), len(events))
        for event in events:
            value = result
            for part in event.payload_ref.path:
                value = value[part]
            self.assertIsNotNone(value)
        self.assertEqual(events, normalize_trace(result))
        self.assertEqual(result, original)
        json.dumps([asdict(event) for event in events])

    def test_search_query_prefers_tool_arguments_over_exported_query(self):
        for fields, expected in (
            ({"args": {"query": "actual"}}, "actual"),
            ({"query": "exported"}, "exported"),
            ({"args": {"query": "actual"}, "query": "exported"}, "actual"),
            ({"args": {"query": " ", "q": "alias"}, "query": "exported"}, "alias"),
            ({"args": {"query": None}, "query": "exported"}, "exported"),
            ({"args": "invalid", "query": []}, None),
        ):
            with self.subTest(fields=fields):
                row = {"trajectory": [{"action": "bm25_search", "observation": "Listing", **fields}]}
                call, result = normalize_trace(row)
                self.assertEqual(call.event_type, TraceEventType.TOOL_CALL)
                self.assertEqual(result.event_type, TraceEventType.TOOL_RESULT)
                self.assertEqual(call.search_query, expected)
                self.assertEqual(result.search_query, expected)
                self.assertEqual(call.tool_call_id, result.tool_call_id)

    def test_control_steps_remain_addressable_without_implying_an_answer(self):
        actions = ["answer", "submit", "stop", "budget", "fix", "fix_rejected", "none", "other"]
        steps = [{"action": action, "observation": "(episode ended)" if action == "other" else "Control"}
                 for action in actions]
        events = normalize_trace({"trajectory": steps, "answer": None})
        self.assertEqual([event.event_type for event in events], [TraceEventType.MODEL_RESPONSE] * len(steps))
        self.assertEqual([event.component_id for event in events], actions)
        self.assertEqual([event.step_index for event in events], list(range(len(steps))))
        self.assertEqual([event.payload_ref.path for event in events],
                         [["trajectory", index] for index in range(len(steps))])
        self.assertTrue(all(event.tool_call_id is None for event in events))

    def test_missing_trace_does_not_invent_steps_or_errors(self):
        for result in ({}, {"trajectory": None}, {"trajectory": [], "status": "error"}):
            with self.subTest(result=result):
                self.assertEqual(normalize_trace(result), [])
        events = normalize_trace({"answer": "", "error": "TimeoutError"})
        self.assertEqual([event.payload_ref.path for event in events], [["answer"], ["error"]])
        self.assertTrue(all(event.step_index is None for event in events))

    def test_partial_and_unknown_entries_remain_addressable(self):
        result = {"trajectory": ["opaque", {}, {"action": "unknown"}, {"action": "visit"}]}
        events = normalize_trace(result)
        self.assertEqual([event.event_type for event in events], [
            TraceEventType.UNKNOWN, TraceEventType.UNKNOWN, TraceEventType.UNKNOWN,
            TraceEventType.TOOL_CALL,
        ])
        self.assertEqual(events[0].payload_ref.path, ["trajectory", 0])
        self.assertEqual(events[2].payload_ref.path, ["trajectory", 2])
        self.assertEqual(events[-1].component_id, "visit")
        self.assertEqual(events[-1].step_index, 3)
        event, = normalize_trace({"trajectory": "opaque"})
        self.assertEqual(event.event_type, TraceEventType.UNKNOWN)

    def test_tool_error_is_preserved_as_an_observation(self):
        result = {"trajectory": [{"action": "bm25_search", "observation": "  ERROR: timeout"}]}
        call, event = normalize_trace(result)
        self.assertEqual(event.event_type, TraceEventType.TOOL_RESULT)
        self.assertEqual(event.error_ref.path, ["trajectory", 0, "observation"])
        self.assertEqual(event.step_index, 0)
        self.assertEqual(event.tool_call_id, call.tool_call_id)
        self.assertIsNone(call.error_ref)
        self.assertEqual(diagnose([result]).reports[0].status, DiagnosisStatus.FAILURE_FOUND)

    def test_normalizes_all_three_checked_in_runner_rows(self):
        path = Path(__file__).resolve().parents[1] / "outputs/evaluation/example/runner_results.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        expected_queries = {
            "q001": ["creator of Python", None],
            "q002": ["Eiffel Tower Paris completed 1889", None],
            "q003": ["closest planet to the Sun", "Mercury closest planet Sun"],
        }
        self.assertEqual([row["id"] for row in rows], list(expected_queries))
        for row in rows:
            with self.subTest(case_id=row["id"]):
                events = normalize_trace(row)
                self.assertEqual([event.event_type for event in events], [
                    TraceEventType.TOOL_CALL, TraceEventType.TOOL_RESULT,
                    TraceEventType.TOOL_CALL, TraceEventType.TOOL_RESULT,
                    TraceEventType.MODEL_RESPONSE, TraceEventType.FINAL_ANSWER,
                ])
                self.assertEqual([event.search_query for event in events[0:4:2]], expected_queries[row["id"]])
                self.assertEqual([event.step_index for event in events], [0, 0, 1, 1, 2, None])
                for call, result in zip(events[0:4:2], events[1:4:2]):
                    self.assertEqual(call.tool_call_id, result.tool_call_id)
                    self.assertEqual(result.payload_ref.path, ["trajectory", call.step_index, "observation"])
                error_paths = [event.error_ref.path for event in events if event.error_ref is not None]
                self.assertEqual(error_paths, [["trajectory", 0, "observation"]] if row["id"] == "q003" else [])
                for event in events:
                    value = row
                    for part in event.payload_ref.path:
                        value = value[part]
                    self.assertIsNotNone(value)


if __name__ == "__main__":
    unittest.main()
