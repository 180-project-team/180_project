"""Exercise the actual Runner-to-Diagnostics boundary with local fake agents."""

import copy
import json
import unittest
from dataclasses import asdict
from tempfile import TemporaryDirectory

from app.diagnostics import DiagnosisStatus, FailureType, SignalLevel, diagnose
from app.runner import AgentRunner, CallableAdapter
from examples.diagnose_runner import run_example


class DiagnosticsIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.run_dir = self.enterContext(TemporaryDirectory())

    def test_example_runs_all_scenarios_through_existing_runner(self):
        rows, batch = run_example(run_dir=self.run_dir)
        self.assertEqual(batch.summary.run_id, "demo-run")
        self.assertEqual(batch.summary.version_id, "demo-v1")
        self.assertEqual(sum(row["status"] == "error" for row in rows), 1)
        self.assertEqual([report.case_id for report in batch.reports],
                         [row["id"] for row in rows])
        self.assertEqual([report.status for report in batch.reports], [
            DiagnosisStatus.NO_FAILURE_DETECTED,
            DiagnosisStatus.FAILURE_FOUND,
            DiagnosisStatus.FAILURE_FOUND,
            DiagnosisStatus.FAILURE_FOUND,
            DiagnosisStatus.NO_FAILURE_DETECTED,
        ])
        self.assertEqual(batch.summary.analyzed_cases, 5)
        self.assertEqual(batch.summary.failed_cases, 3)
        self.assertEqual(batch.summary.failure_counts, {
            FailureType.OUTPUT_FORMAT_ERROR: 1,
            FailureType.UNKNOWN: 1,
            FailureType.TOOL_EXECUTION_FAILURE: 1,
        })
        tool_report = batch.reports[3]
        self.assertIsNone(tool_report.critical_step)
        self.assertEqual(tool_report.signals[0].step_index, 0)
        self.assertEqual(batch.reports[4].signals[0].level, SignalLevel.WARNING)
        original = copy.deepcopy(rows)
        diagnose(rows)
        self.assertEqual(rows, original)
        for row, report in zip(rows, batch.reports):
            for ref in report.evidence_refs:
                value = row
                for part in ref.path:
                    value = value[part]
        json.dumps(asdict(batch))

    def test_diagnosis_handles_runner_error_without_trajectory(self):
        class ErrorAgent:
            def ask(self, question):
                return {"error": "Synthetic failure", "trajectory": [
                    {"action": "bm25_search", "observation": "ERROR: Timeout"}
                ], "traceback": "Synthetic traceback"}
        runner = AgentRunner(lambda _: CallableAdapter(ErrorAgent))
        row, = runner.run([{"question": "question without dataset identifier"}], self.run_dir)
        self.assertEqual(row["id"], "1")
        self.assertEqual(row["status"], "error")
        self.assertIsNone(row["answer"])
        self.assertNotIn("trajectory", row)
        self.assertNotIn("traceback", row)
        report, = diagnose([row]).reports
        self.assertEqual(report.case_id, row["id"])
        self.assertEqual(report.failure_type, FailureType.UNKNOWN)
        self.assertIsNone(report.critical_step)
        self.assertEqual([signal.rule_id for signal in report.signals], ["execution_error"])


if __name__ == "__main__":
    unittest.main()
