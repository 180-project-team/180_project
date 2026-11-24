"""Test the file-based Diagnostics entry point and report publication."""

import json
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import call, create_autospec, patch

from app.diagnostics import DiagnosisStatus, FailureType, QualityThresholds, diagnose_run
from app.llm import LLMService


class DiagnoseRunTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.output = self.root / "diagnostics"
        self.examples = Path(__file__).resolve().parents[1] / "outputs" / "evaluation" / "example"
        self.runner = self.examples / "runner_results.jsonl"
        self.evaluation = self.examples / "results.jsonl"

    def run_diagnostics(self, **options):
        return diagnose_run(**{
            "run_results_path": self.runner,
            "evaluation_results_path": self.evaluation,
            "output_dir": self.output,
            **options,
        })

    def write_rows(self, path, rows):
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    def reports(self, output=None):
        path = self.output if output is None else Path(output)
        return [json.loads(line) for line in (path / "diagnoses.jsonl").read_text(encoding="utf-8").splitlines()]

    def summary(self, output=None):
        path = self.output if output is None else Path(output)
        return json.loads((path / "summary.json").read_text(encoding="utf-8"))

    def test_writes_exactly_the_two_public_models_in_runner_order(self):
        input_bytes = [path.read_bytes() for path in (self.runner, self.evaluation)]
        output = self.output / "nested"
        batch = self.run_diagnostics(
            run_results_path=str(self.runner), output_dir=str(output), run_id="run-1", version_id="agent-v1",
        )

        self.assertEqual({path.name for path in output.iterdir()}, {"diagnoses.jsonl", "summary.json"})
        records = self.reports(output)
        self.assertEqual([report["case_id"] for report in records], ["q001", "q002", "q003"])
        self.assertEqual(records, [asdict(report) for report in batch.reports])
        self.assertEqual(self.summary(output), asdict(batch.summary))
        self.assertEqual(batch.summary.analyzed_cases, len(records))
        self.assertEqual(batch.summary.failed_cases, 1)
        for report in records:
            self.assertEqual((report["run_id"], report["version_id"]), ("run-1", "agent-v1"))
            self.assertIsNone(report["classifier_attempt"])
        failed = records[-1]
        self.assertEqual(failed["failure_type"], "TOOL_EXECUTION_FAILURE")
        self.assertEqual(failed["status"], "FAILURE_FOUND")
        self.assertEqual(failed["signals"][0]["level"], "FAILURE")
        self.assertEqual(failed["hypotheses"][0]["scope"], "STRUCTURAL")
        self.assertEqual(failed["evidence_refs"][0], {"path": ["trajectory", 0, "observation"]})
        self.assertEqual([path.read_bytes() for path in (self.runner, self.evaluation)], input_bytes)

    def test_injects_mock_service_and_forwards_quality_thresholds(self):
        runner = self.root / "runner.jsonl"
        evaluator = self.root / "evaluator.jsonl"
        self.write_rows(runner, [{
            "id": "q1", "question": "Кто режиссёр каждого из двух фильмов?", "answer": "Алиса", "status": "ok",
            "trajectory": [{"action": "visit", "observation": "Алиса сняла первый фильм. Боб снял второй."}],
        }])
        self.write_rows(evaluator, [{"id": "q1", "status": "ok", "fact_recall": 0.85, "retrieval_fact_recall": 1.0}])
        service = create_autospec(LLMService, instance=True, spec_set=True)
        service.chat.return_value = '{"failure_type":"INCOMPLETE_ANSWER"}'

        batch = self.run_diagnostics(
            run_results_path=runner, evaluation_results_path=evaluator, llm_service=service,
            quality_thresholds=QualityThresholds(fact_recall=0.9),
        )

        report, = batch.reports
        self.assertEqual(report.failure_type, FailureType.INCOMPLETE_ANSWER)
        self.assertEqual(report.classifier_attempt.status, "accepted")
        messages, = service.chat.call_args.args
        self.assertEqual(service.mock_calls, [call.chat(messages, temperature=0)])
        self.assertIn("Боб снял второй", messages[1]["content"])
        record, = self.reports()
        self.assertEqual(record, asdict(report))
        self.assertEqual(set(record), {
            "case_id", "run_id", "version_id", "status", "failure_type", "critical_step",
            "evidence_refs", "rationale", "confidence", "suggested_scope", "signals",
            "hypotheses", "classifier_attempt",
        })
        self.assertEqual(record["status"], "FAILURE_FOUND")
        self.assertEqual(record["failure_type"], "INCOMPLETE_ANSWER")
        self.assertEqual(record["confidence"], "low")
        self.assertEqual(record["suggested_scope"], "PROMPT")
        self.assertEqual(record["classifier_attempt"], {"status": "accepted", "cost_usd": None})
        self.assertEqual(record["evidence_refs"], [
            {"path": ["metrics", "fact_recall"]},
            {"path": ["metrics", "retrieval_fact_recall"]},
            {"path": ["trajectory", 0, "observation"]},
        ])
        self.assertEqual([item["scope"] for item in record["hypotheses"]], ["PROMPT", "PROMPT", "STRUCTURAL"])
        self.assertEqual(record["hypotheses"][0]["hypothesis_id"], "check_required_answer_facts")
        for hypothesis in record["hypotheses"]:
            self.assertEqual(set(hypothesis), {"hypothesis_id", "scope", "advice"})
            self.assertTrue(hypothesis["advice"].strip())
        self.assertEqual(record["signals"][-1]["rule_id"], "classifier_selection")
        self.assertIsNone(record["signals"][-1]["component_id"])
        self.assertIsNone(record["critical_step"])
        self.assertEqual(self.summary(), asdict(batch.summary))

    def test_serializes_secondary_prompt_hypotheses_and_unknown_scope(self):
        runner = self.root / "runner.jsonl"
        evaluator = self.root / "evaluator.jsonl"
        case = {"question": "Who directed both films?", "answer": "Alice", "status": "ok"}
        self.write_rows(runner, [
            {**case, "id": "mixed", "trajectory": [
                {"action": "bm25_search", "observation": "ERROR: Timeout"},
                {"action": "visit", "observation": "Alice directed the first film. Bob directed the second."},
            ]},
            {**case, "id": "uncertain", "trajectory": [{"action": "answer"}]},
        ])
        self.write_rows(evaluator, [
            {"id": case_id, "status": "ok", "fact_recall": 0.5, "retrieval_fact_recall": 1.0}
            for case_id in ("mixed", "uncertain")
        ])
        service = create_autospec(LLMService, instance=True, spec_set=True)
        service.chat.return_value = '{"failure_type":"INCOMPLETE_ANSWER"}'

        batch = self.run_diagnostics(run_results_path=runner, evaluation_results_path=evaluator, llm_service=service)

        mixed, uncertain = self.reports()
        self.assertEqual([mixed, uncertain], [asdict(report) for report in batch.reports])
        self.assertEqual((mixed["failure_type"], mixed["suggested_scope"]), ("TOOL_EXECUTION_FAILURE", "STRUCTURAL"))
        self.assertEqual(mixed["signals"][0]["component_id"], "bm25_search")
        self.assertEqual(mixed["signals"][0]["evidence_refs"], [{"path": ["trajectory", 0, "observation"]}])
        self.assertEqual(mixed["signals"][-1]["failure_type"], "INCOMPLETE_ANSWER")
        self.assertIsNone(mixed["signals"][-1]["component_id"])
        self.assertTrue(any(item["scope"] == "PROMPT" for item in mixed["hypotheses"]))
        self.assertEqual((uncertain["failure_type"], uncertain["suggested_scope"]), ("UNKNOWN", "UNKNOWN"))
        for field in ("run_id", "version_id", "critical_step", "classifier_attempt"):
            self.assertIsNone(uncertain[field])
        service.chat.assert_called_once()

    def test_repeat_replaces_reports_and_preserves_summary_patterns(self):
        runner = self.root / "runner.jsonl"
        evaluator = self.root / "evaluator.jsonl"
        self.write_rows(runner, [
            {"id": case_id, "question": "Who?", "answer": None, "status": "error", "error": "Timeout"}
            for case_id in ("b", "a")
        ])
        self.write_rows(evaluator, [{"id": case_id, "status": "error"} for case_id in ("a", "b")])

        self.run_diagnostics(run_results_path=runner, evaluation_results_path=evaluator, run_id="first")
        batch = self.run_diagnostics(run_results_path=runner, evaluation_results_path=evaluator, run_id="second")

        records = self.reports()
        self.assertEqual([report["case_id"] for report in records], ["b", "a"])
        self.assertEqual([report["run_id"] for report in records], ["second", "second"])
        self.assertEqual(self.summary(), asdict(batch.summary))
        pattern = next(item for item in self.summary()["patterns"] if item["rule_id"] == "execution_error")
        self.assertEqual(pattern["level"], "FAILURE")
        self.assertEqual(pattern["report_indices"], [0, 1])
        self.assertEqual(pattern["run_id"], "second")

    def test_invalid_input_does_not_create_or_update_reports(self):
        invalid = self.root / "invalid.jsonl"
        invalid.write_text("{", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            self.run_diagnostics(evaluation_results_path=invalid)
        self.assertFalse(self.output.exists())

        self.run_diagnostics()
        initial = {path.name: path.read_bytes() for path in self.output.iterdir()}
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            self.run_diagnostics(evaluation_results_path=invalid)
        self.assertEqual({path.name: path.read_bytes() for path in self.output.iterdir()}, initial)

    def test_rejects_overlapping_input_and_output_paths(self):
        self.output.mkdir()
        runner = self.output / "diagnoses.jsonl"
        evaluator = self.output / "summary.json"
        runner.write_bytes(self.runner.read_bytes())
        evaluator.write_bytes(self.evaluation.read_bytes())
        for options in ({"run_results_path": runner}, {"evaluation_results_path": evaluator}):
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, "overlaps input artifact"):
                    self.run_diagnostics(**options)
                self.assertEqual(runner.read_bytes(), self.runner.read_bytes())
                self.assertEqual(evaluator.read_bytes(), self.evaluation.read_bytes())

    def test_rejects_a_hard_link_to_an_input_artifact(self):
        runner = self.root / "runner.jsonl"
        runner.write_bytes(self.runner.read_bytes())
        self.output.mkdir()
        output = self.output / "diagnoses.jsonl"
        output.hardlink_to(runner)

        with self.assertRaisesRegex(ValueError, "overlaps input artifact"):
            self.run_diagnostics(run_results_path=runner)
        self.assertTrue(output.samefile(runner))
        self.assertEqual(runner.read_bytes(), self.runner.read_bytes())
        self.assertFalse((self.output / "summary.json").exists())

    def test_write_failure_is_visible_and_preserves_existing_reports(self):
        batch = self.run_diagnostics(run_id="first")
        with patch.object(Path, "write_text", side_effect=OSError("summary write failed")):
            with self.assertRaisesRegex(OSError, "summary write failed"):
                self.run_diagnostics(run_id="second")

        self.assertEqual({path.name for path in self.output.iterdir()}, {"diagnoses.jsonl", "summary.json"})
        self.assertEqual(self.reports(), [asdict(report) for report in batch.reports])
        self.assertEqual(self.summary(), asdict(batch.summary))
