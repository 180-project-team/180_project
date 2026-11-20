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
        self.runs_root = self.enterContext(TemporaryDirectory())

    def test_example_runs_all_scenarios_through_existing_runner(self):
        run_report, batch = run_example(runs_root=self.runs_root)
        rows = run_report.load_results()
        self.assertEqual(batch.summary.run_id, run_report.run_id)
        self.assertEqual(run_report.failed_count, 1)
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
        run_report = AgentRunner(lambda: CallableAdapter(ErrorAgent), runs_root=self.runs_root).run([
            {"question": "question without dataset identifier"},
        ])
        row, = run_report.load_results()
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
