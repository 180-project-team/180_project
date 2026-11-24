"""Check the JSONL boundary and evidence paths of joined diagnostic cases."""

import json
import unittest
from contextlib import chdir
from pathlib import Path
from tempfile import TemporaryDirectory

from app.diagnostics import DiagnosisStatus, EvidenceRef, FailureType, diagnose, load_diagnostic_cases


def runner_row(case_id, **fields):
    return {"id": case_id, "question": "Кто?", "answer": "Alice", "status": "ok", **fields}


def evaluation_row(case_id, **fields):
    return {"id": case_id, "status": "ok", **fields}


class DiagnosticArtifactTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.runner = self.root / "runner.jsonl"
        self.evaluator = self.root / "evaluation.jsonl"

    def write(self, path, rows):
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    def load(self):
        return load_diagnostic_cases(self.runner, self.evaluator)

    def assert_location(self, error, path, line_number):
        self.assertIn(f"{path}:{line_number}", str(error))

    def test_main_examples_join_and_diagnose_from_another_directory(self):
        examples = Path(__file__).resolve().parents[1] / "outputs" / "evaluation" / "example"
        runner_path, evaluation_path = (examples / name for name in ("runner_results.jsonl", "results.jsonl"))
        original = [path.read_bytes() for path in (runner_path, evaluation_path)]

        with chdir(self.root):
            cases = load_diagnostic_cases(str(runner_path), evaluation_path)
            batch = diagnose(cases)

        self.assertEqual([case["case_id"] for case in cases], ["q001", "q002", "q003"])
        self.assertEqual([report.case_id for report in batch.reports], ["q001", "q002", "q003"])
        self.assertEqual([report.failure_type for report in batch.reports], [
            FailureType.UNKNOWN, FailureType.UNKNOWN, FailureType.TOOL_EXECUTION_FAILURE,
        ])
        self.assertEqual(cases[0]["metrics"], {
            "fact_recall": 1.0, "fact_precision": 1.0, "retrieval_fact_recall": 1.0, "faithfulness": 1.0,
        })
        self.assertEqual(cases[2]["execution_metrics"]["failed_tool_calls_count"], 1)
        for case, raw in zip(cases, original[0].decode("utf-8").splitlines(), strict=True):
            self.assertEqual(case["trajectory"], json.loads(raw)["trajectory"])
        self.assertEqual([path.read_bytes() for path in (runner_path, evaluation_path)], original)

    def test_normalized_ids_join_in_runner_order_and_keep_source_fields_separate(self):
        trajectory = [None, {"action": "visit", "observation": "Document excerpt", "read_ids": ["document"]}]
        self.write(self.runner, [
            runner_row(" 9 ", trajectory=trajectory, duration_ms=12, stopped="answer", usage={"total_tokens": 7}),
            runner_row(8, trajectory=None),
        ])
        self.write(self.evaluator, [
            evaluation_row(" 8 ", status="error", fact_recall=None),
            evaluation_row(9, duration_ms=99, tool_calls_count=2, fact_recall=0.0, fact_precision=None,
                           faithfulness="invalid score", question="Evaluation extension"),
        ])
        original = self.runner.read_bytes(), self.evaluator.read_bytes()

        scored, unavailable = self.load()

        self.assertEqual([scored["case_id"], unavailable["case_id"]], ["9", "8"])
        self.assertEqual(scored["question"], "Кто?")
        self.assertEqual(scored["trajectory"], trajectory)
        self.assertEqual(scored["usage"], {"total_tokens": 7})
        self.assertEqual(scored["stopped"], "answer")
        self.assertEqual(scored["duration_ms"], 12)
        self.assertEqual(scored["execution_metrics"], {"duration_ms": 99, "tool_calls_count": 2})
        self.assertEqual(scored["metrics"], {"fact_recall": 0.0, "fact_precision": None, "faithfulness": "invalid score"})
        self.assertIs(type(scored["metrics"]["fact_recall"]), float)
        self.assertEqual((unavailable["status"], unavailable["evaluation_status"]), ("ok", "error"))
        self.assertEqual(unavailable["metrics"], {"fact_recall": None})
        self.assertEqual(unavailable["execution_metrics"], {})
        self.assertEqual((self.runner.read_bytes(), self.evaluator.read_bytes()), original)

    def test_evidence_resolves_to_the_correct_source_fields(self):
        observation = "ERROR: Search timed out"
        self.write(self.runner, [
            runner_row("scored", trajectory=[{"action": "bm25_search", "observation": observation}]),
            runner_row("unscored"),
        ])
        self.write(self.evaluator, [
            evaluation_row("unscored", status="error"),
            evaluation_row("scored", fact_recall=0.25, faithfulness=True),
        ])
        cases = self.load()
        scored, unscored = diagnose(cases).reports
        self.assertEqual(scored.failure_type, FailureType.TOOL_EXECUTION_FAILURE)
        self.assertEqual(unscored.status, DiagnosisStatus.INSUFFICIENT_DATA)
        for case, report in zip(cases, (scored, unscored), strict=True):
            for ref in report.evidence_refs + [ref for signal in report.signals for ref in signal.evidence_refs]:
                value = case
                for part in ref.path:
                    value = value[part]
        self.assertIn(EvidenceRef(["trajectory", 0, "observation"]), scored.evidence_refs)
        self.assertEqual(cases[0]["trajectory"][0]["observation"], observation)
        self.assertIn(EvidenceRef(["metrics", "fact_recall"]), scored.evidence_refs)
        self.assertEqual(cases[0]["metrics"]["fact_recall"], 0.25)
        self.assertIn("invalid_quality_metrics", [signal.rule_id for signal in scored.signals])
        self.assertEqual(unscored.evidence_refs, [EvidenceRef(["evaluation_status"])])
        self.assertEqual(cases[1]["evaluation_status"], "error")
        self.assertNotIn("trajectory", cases[1])

    def test_ids_must_be_present_and_valid_in_either_artifact(self):
        for path, make_row in ((self.runner, runner_row), (self.evaluator, evaluation_row)):
            for raw_id in (None, "", "  ", True, 4.0, [], {}):
                with self.subTest(path=path.name, raw_id=raw_id):
                    self.write(self.runner, [runner_row("valid")])
                    self.write(self.evaluator, [evaluation_row("valid")])
                    row = make_row(raw_id)
                    if raw_id is None:
                        del row["id"]
                    self.write(path, [row])
                    with self.assertRaisesRegex(ValueError, "non-empty string or integer id") as error:
                        self.load()
                    self.assert_location(error.exception, path, 1)

    def test_detects_duplicate_ids_after_normalization_in_either_artifact(self):
        for path, make_row in ((self.runner, runner_row), (self.evaluator, evaluation_row)):
            with self.subTest(path=path.name):
                self.write(self.runner, [runner_row("4")])
                self.write(self.evaluator, [evaluation_row("4")])
                self.write(path, [make_row(4), make_row(" 4 ")])
                with self.assertRaisesRegex(ValueError, "duplicate id '4'.*first on line 1") as error:
                    self.load()
                self.assert_location(error.exception, path, 2)

    def test_mismatched_ids_report_the_missing_tasks_on_both_sides(self):
        self.write(self.runner, [runner_row("common"), runner_row("runner-only")])
        self.write(self.evaluator, [evaluation_row("evaluator-only"), evaluation_row("common")])
        with self.assertRaises(ValueError) as error:
            self.load()
        self.assertIn(f"Missing in Runner ({self.runner}): ['evaluator-only']", str(error.exception))
        self.assertIn(f"Missing in Evaluator ({self.evaluator}): ['runner-only']", str(error.exception))

    def test_empty_and_missing_files_report_the_input_path(self):
        for path in (self.runner, self.evaluator):
            with self.subTest(path=path.name):
                self.write(self.runner, [runner_row("a")])
                self.write(self.evaluator, [evaluation_row("a")])
                for content in ("", "\n  \n"):
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "expected at least one task") as error:
                        self.load()
                    self.assertIn(str(path), str(error.exception))
                path.unlink()
                with self.assertRaises(FileNotFoundError) as error:
                    self.load()
                self.assertEqual(error.exception.filename, str(path))

    def test_json_and_encoding_errors_use_physical_line_numbers(self):
        for text, description in ((b"{", "invalid JSON"), (b"[]", "expected a JSON object"),
                                  (b'{"value":NaN}', "invalid JSON"), (b"\xff", "invalid UTF-8")):
            with self.subTest(text=text):
                self.write(self.runner, [runner_row("a")])
                line = json.dumps(evaluation_row("a")).encode("utf-8")
                self.evaluator.write_bytes(line + b"\n \n" + text + b"\n")
                with self.assertRaisesRegex(ValueError, description) as error:
                    self.load()
                self.assert_location(error.exception, self.evaluator, 3)

    def test_structure_errors_and_reserved_fields_are_explicit(self):
        for fields, description in (
            ({"status": "unknown"}, "expected status"),
            ({"question": " "}, "non-empty question"),
            ({"trajectory": {}}, "expected trajectory"),
            ({"metrics": {}}, "reserved join fields.*metrics"),
        ):
            with self.subTest(fields=fields):
                self.write(self.runner, [runner_row("a", **fields)])
                self.write(self.evaluator, [evaluation_row("a")])
                with self.assertRaisesRegex(ValueError, description) as error:
                    self.load()
                self.assert_location(error.exception, self.runner, 1)
        self.write(self.runner, [runner_row("a")])
        self.write(self.evaluator, [evaluation_row("a", status=None)])
        with self.assertRaisesRegex(ValueError, "expected status") as error:
            self.load()
        self.assert_location(error.exception, self.evaluator, 1)
