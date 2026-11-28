"""Report selection and batch aggregation without external agents or services."""

import copy
import json
import unittest
from dataclasses import asdict

from app.diagnostics import (
    DiagnosisStatus, FailureType, SignalLevel, SuggestedScope, diagnose,
    normalize_trace,
)


def tool_step(error=None):
    return {
        "action": "bm25_search",
        "observation": f"ERROR: {error}" if error else "Results",
    }


class DiagnosisReportTests(unittest.TestCase):
    def test_mixed_batch_counts_paths_serialization_and_immutability(self):
        rows = [
            {"id": "ok", "status": "ok", "answer": "answer"},
            {"id": "missing", "status": "ok"},
            {"id": "error", "status": "error", "error": "RuntimeError"},
            {},
        ]
        original = copy.deepcopy(rows)
        batch = diagnose(iter(rows), run_id="run", version_id="v1")
        self.assertEqual(rows, original)
        self.assertEqual([report.status for report in batch.reports], [
            DiagnosisStatus.NO_FAILURE_DETECTED, DiagnosisStatus.FAILURE_FOUND,
            DiagnosisStatus.FAILURE_FOUND, DiagnosisStatus.INSUFFICIENT_DATA,
        ])
        self.assertEqual(batch.summary.analyzed_cases, 4)
        self.assertEqual((batch.summary.run_id, batch.summary.version_id), ("run", "v1"))
        self.assertEqual(batch.summary.failed_cases, 2)
        self.assertEqual(batch.summary.failure_counts,
                         {FailureType.OUTPUT_FORMAT_ERROR: 1, FailureType.UNKNOWN: 1})
        self.assertEqual(batch.summary.report_indices, [0, 1, 2, 3])
        self.assertEqual(batch.reports[1].suggested_scope, SuggestedScope.PROMPT)
        self.assertEqual(batch.reports[2].suggested_scope, SuggestedScope.UNKNOWN)
        for row, report in zip(rows, batch.reports):
            for ref in report.evidence_refs:
                value = row
                for part in ref.path:
                    value = value[part]
        decoded = json.loads(json.dumps(asdict(batch)))
        self.assertEqual(decoded["summary"]["failure_counts"]["OUTPUT_FORMAT_ERROR"], 1)

    def test_terminal_tool_error_has_candidate_step_and_all_signals(self):
        row = {"status": "error", "error": "failure",
               "trajectory": [{"action": "none"}, tool_step("Timeout")]}
        batch = diagnose([row])
        report = batch.reports[0]
        self.assertEqual(report.failure_type, FailureType.TOOL_EXECUTION_FAILURE)
        self.assertEqual(report.critical_step, 1)
        self.assertEqual(report.suggested_scope, SuggestedScope.STRUCTURAL)
        self.assertEqual([signal.rule_id for signal in report.signals], ["execution_error", "tool_error"])
        self.assertEqual(batch.summary.failed_cases, 1)
        self.assertEqual(sum(batch.summary.failure_counts.values()), 1)
        self.assertIn("not a proven root cause", report.rationale)

    def test_success_after_tool_error_does_not_claim_critical_step(self):
        row = {"status": "ok", "answer": "42",
               "trajectory": [tool_step("Timeout"), tool_step()]}
        report = diagnose([row]).reports[0]
        self.assertEqual(report.status, DiagnosisStatus.FAILURE_FOUND)
        self.assertIsNone(report.critical_step)
        self.assertEqual(len(report.signals), 1)

    def test_ambiguous_errors_and_later_events_have_no_critical_step(self):
        for trace in ([tool_step("Timeout"), tool_step("Timeout")],
                      [tool_step("Timeout"), {"action": "none"}],
                      [tool_step("Timeout"), tool_step()]):
            with self.subTest(trace=trace):
                report = diagnose([{"status": "error", "trajectory": trace}]).reports[0]
                self.assertIsNone(report.critical_step)

    def test_warnings_are_retained_but_not_counted_as_failures(self):
        rows = [{"id": "duplicate-id", "status": "ok", "answer": "42",
                 "trajectory": [{"action": "bm25_search", "query": "q"} for _ in range(2)]} for _ in range(2)]
        batch = diagnose(rows)
        for report in batch.reports:
            self.assertEqual(report.status, DiagnosisStatus.NO_FAILURE_DETECTED)
            self.assertEqual(report.failure_type, FailureType.UNKNOWN)
            self.assertIsNone(report.critical_step)
            self.assertEqual(report.signals[0].level, SignalLevel.WARNING)
        self.assertEqual(batch.summary.failed_cases, 0)
        pattern, = batch.summary.patterns
        self.assertEqual(pattern.level, SignalLevel.WARNING)
        self.assertEqual(pattern.report_indices, [0, 1])

    def test_patterns_stay_within_run_and_version(self):
        rows = [{"status": "ok", "run_id": run, "version_id": version}
                for run, version in (("a", "v1"), ("b", "v1"), ("a", "v2"), ("a", "v1"))]
        batch = diagnose(rows)
        pattern, = batch.summary.patterns
        self.assertIsNone(batch.summary.run_id)
        self.assertIsNone(batch.summary.version_id)
        self.assertEqual((pattern.run_id, pattern.version_id), ("a", "v1"))
        self.assertEqual(pattern.report_indices, [0, 3])
        self.assertEqual(batch.summary.failure_counts, {FailureType.OUTPUT_FORMAT_ERROR: 4})
        overridden = diagnose(rows, run_id="override", version_id="v3")
        self.assertEqual(overridden.summary.patterns[0].report_indices, [0, 1, 2, 3])

    def test_normalizer_failure_is_isolated_and_not_an_agent_failure(self):
        def normalizer(row):
            if row.get("id") == "broken":
                raise ValueError("unreadable trace")
            return normalize_trace(row)
        batch = diagnose([{"id": "broken"}, {"status": "ok", "answer": "42"}], normalizer=normalizer)
        self.assertEqual(batch.reports[0].status, DiagnosisStatus.DIAGNOSTIC_ERROR)
        self.assertIn("ValueError: unreadable trace", batch.reports[0].rationale)
        self.assertEqual(batch.reports[0].evidence_refs, [])
        self.assertEqual(batch.reports[1].status, DiagnosisStatus.NO_FAILURE_DETECTED)
        self.assertEqual(batch.summary.failed_cases, 0)

    def test_validator_controls_known_contract_and_errors_are_isolated(self):
        row = {"status": "ok", "answer": "text"}
        self.assertEqual(diagnose([row], answer_validator=lambda value: False).reports[0].failure_type,
                         FailureType.OUTPUT_FORMAT_ERROR)
        def broken(value):
            raise RuntimeError("validator unavailable")
        report = diagnose([row], answer_validator=broken).reports[0]
        self.assertEqual(report.status, DiagnosisStatus.DIAGNOSTIC_ERROR)

    def test_unrecognized_metrics_do_not_invent_semantic_failures(self):
        row = {"status": "ok", "answer": "text", "metrics": {"unknown_score": 0}}
        batch = diagnose([row])
        self.assertEqual(batch.reports[0].status, DiagnosisStatus.NO_FAILURE_DETECTED)

    def test_empty_batch_has_empty_summary(self):
        batch = diagnose([])
        self.assertEqual(batch.summary.analyzed_cases, 0)
        self.assertEqual(batch.summary.failed_cases, 0)
        self.assertEqual(batch.summary.failure_counts, {})
        self.assertEqual(batch.summary.patterns, [])
        self.assertEqual(batch.summary.report_indices, [])

    def test_invalid_rows_are_caller_errors(self):
        with self.assertRaisesRegex(TypeError, "mapping"):
            diagnose([None])


if __name__ == "__main__":
    unittest.main()
