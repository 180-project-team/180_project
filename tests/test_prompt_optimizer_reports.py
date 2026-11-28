"""Exercise the optimizer's reader against real Diagnostics artifacts."""

import json
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

from app.diagnostics import diagnose, diagnose_run
from app.prompt_optimizer.diagnostic_report import load_diagnosis_reports


class DiagnosticReportReaderTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))

    def report(self):
        return asdict(diagnose([{
            "id": "q1", "question": "Who directed the film?", "status": "ok",
            "answer": "Alice", "trajectory": [],
        }], run_id="run-1", version_id="v1").reports[0])

    def test_reads_actual_diagnose_run_output_without_modifying_inputs(self):
        examples = Path(__file__).resolve().parents[1] / "outputs" / "evaluation" / "example"
        inputs = [examples / "runner_results.jsonl", examples / "results.jsonl"]
        original = [path.read_bytes() for path in inputs]
        batch = diagnose_run(
            run_results_path=inputs[0], evaluation_results_path=inputs[1],
            output_dir=self.root, run_id="run-1", version_id="v1",
        )

        reports = load_diagnosis_reports(self.root / "diagnoses.jsonl")

        self.assertEqual(reports, [asdict(report) for report in batch.reports])
        self.assertEqual(len(reports), 3)
        self.assertEqual([path.read_bytes() for path in inputs], original)
        with self.assertRaisesRegex(ValueError, "DiagnosisReport"):
            load_diagnosis_reports(self.root / "summary.json")

    def test_single_pretty_json_preserves_nulls_and_secondary_prompt_hypotheses(self):
        report = self.report()
        report["suggested_scope"] = "STRUCTURAL"
        report["hypotheses"] = [
            {"hypothesis_id": "retry", "scope": "STRUCTURAL", "advice": "Retry failed tools."},
            {"hypothesis_id": "verify", "scope": "PROMPT", "advice": "Check evidence before answering."},
        ]
        source = self.root / "report.json"
        source.write_text(json.dumps(report, indent=2), encoding="utf-8")

        self.assertEqual(load_diagnosis_reports(str(source)), [report])
        self.assertIsNone(load_diagnosis_reports(source)[0]["critical_step"])

    def test_jsonl_preserves_runner_order_and_ignores_blank_lines(self):
        first = self.report()
        second = {**first, "case_id": "q2"}
        source = self.root / "reports.jsonl"
        source.write_text(json.dumps(second) + "\n\n" + json.dumps(first) + "\n", encoding="utf-8")

        self.assertEqual(load_diagnosis_reports(source), [second, first])

    def test_corrupt_later_record_rejects_the_entire_input_with_line_number(self):
        source = self.root / "reports.jsonl"
        source.write_text(json.dumps(self.report()) + "\n\n{broken}\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "line 3: invalid JSON"):
            load_diagnosis_reports(source)

    def test_rejects_empty_reports_and_malformed_hypotheses(self):
        source = self.root / "reports.jsonl"
        source.write_text("\n\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "empty"):
            load_diagnosis_reports(source)
        report = {**self.report(), "hypotheses": [{"scope": "PROMPT", "advice": "Check evidence."}]}
        source.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid diagnostic hypothesis"):
            load_diagnosis_reports(source)


if __name__ == "__main__":
    unittest.main()
