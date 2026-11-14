"""Contract tests for the initial diagnostics interface."""

import copy
import json
import unittest
from dataclasses import asdict

from app.diagnostics import (
    Confidence, DiagnosisStatus, FailureType, SuggestedScope, diagnose,
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
        for report in batch.reports:
            self.assertEqual((report.run_id, report.version_id), ("run", "v1"))
            self.assertEqual(report.status, DiagnosisStatus.INSUFFICIENT_DATA)
            self.assertEqual(report.failure_type, FailureType.UNKNOWN)
            self.assertEqual(report.confidence, Confidence.LOW)
            self.assertEqual(report.suggested_scope, SuggestedScope.UNKNOWN)
            self.assertIsNone(report.critical_step)
            self.assertEqual(report.evidence_refs, [])

    def test_empty_batch(self):
        self.assertEqual(diagnose([]).reports, [])

    def test_result_is_json_serializable(self):
        batch = diagnose([{"id": "case"}])
        decoded = json.loads(json.dumps(asdict(batch)))
        self.assertEqual(decoded["reports"][0]["status"], "INSUFFICIENT_DATA")
        self.assertIsNone(decoded["reports"][0]["run_id"])


if __name__ == "__main__":
    unittest.main()
