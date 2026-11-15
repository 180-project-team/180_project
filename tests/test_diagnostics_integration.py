"""Exercise the actual Runner-to-Diagnostics boundary with local fake agents."""

import copy
import json
import unittest
from dataclasses import asdict

from app.diagnostics import DiagnosisStatus, FailureType, SignalLevel, diagnose
from app.runner import AgentRunner, CallableAdapter
from examples.diagnose_runner import run_example


class DiagnosticsIntegrationTests(unittest.TestCase):
    def test_example_runs_all_scenarios_through_existing_runner(self):
        run_report, batch = run_example()
        self.assertEqual(run_report.failed_count, 1)
        self.assertEqual([report.case_id for report in batch.reports],
                         [row["id"] for row in run_report.results])
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
        original = copy.deepcopy(run_report.results)
        diagnose(run_report.results)
        self.assertEqual(run_report.results, original)
        for row, report in zip(run_report.results, batch.reports):
            for ref in report.evidence_refs:
                value = row
                for part in ref.path:
                    value = value[part]
        json.dumps(asdict(batch))

    def test_runner_error_response_does_not_preserve_raw_trajectory(self):
        class ErrorAgent:
            def ask(self, question):
                return {"error": "Synthetic failure", "trajectory": [
                    {"role": "assistant", "tool_results": [{"error": "Timeout"}]}
                ]}
        run_report = AgentRunner(lambda: CallableAdapter(ErrorAgent)).run([
            {"question": "question without dataset identifier"},
        ])
        self.assertNotIn("trajectory", run_report.results[0])
        report, = diagnose(run_report.results).reports
        self.assertEqual(report.case_id, run_report.results[0]["id"])
        self.assertEqual(report.failure_type, FailureType.UNKNOWN)
        self.assertIsNone(report.critical_step)
        self.assertEqual([signal.rule_id for signal in report.signals], ["execution_error"])


if __name__ == "__main__":
    unittest.main()
