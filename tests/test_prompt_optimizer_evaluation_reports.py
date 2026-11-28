"""Missing scores from the shared Evaluator must never become measured zeroes."""

import json
import math
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from app.prompt_optimizer.evaluation_report import EvaluationReport, load_evaluation_report
from app.prompt_optimizer.policy import SelectionPolicy


class EvaluationReportTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.runner = self.root / "runner.jsonl"
        self.evaluator = self.root / "evaluator.jsonl"

    def load(self, scores, statuses=None):
        statuses = statuses or ["ok"] * len(scores)
        self.runner.write_text("".join(json.dumps({"id": f"q{i}", "status": "ok", "answer": "answer"}) + "\n"
                                       for i in range(len(scores))), encoding="utf-8")
        self.evaluator.write_text("".join(json.dumps({"id": f"q{i}", "status": status, "fact_recall": value}) + "\n"
                                          for i, (value, status) in enumerate(zip(scores, statuses))), encoding="utf-8")
        return load_evaluation_report(self.runner, self.evaluator, prompt_version_id="v1", split="validation")

    def test_ok_with_null_is_preserved_and_does_not_create_an_aggregate_score(self):
        report = self.load([0.0, None, 1.0])
        self.assertEqual([item.score for item in report.items], [0.0, None, 1.0])
        self.assertEqual(report.scored_tasks, 2)
        self.assertIsNone(report.score)
        self.assertIsNone(report.items[0].latency_ms)
        serialized = json.loads(json.dumps(report.to_dict()))
        self.assertIsNone(serialized["items"][1]["score"])
        self.assertIsNone(serialized["score"])
        self.assertEqual((serialized["scored_tasks"], serialized["total_tasks"]), (2, 3))

    def test_absent_metric_and_evaluation_error_are_unknown(self):
        report = self.load([None], ["error"])
        self.assertIsNone(report.items[0].score)
        self.assertIsNone(report.score)
        self.evaluator.write_text('{"id":"q0","status":"ok"}\n', encoding="utf-8")
        report = load_evaluation_report(self.runner, self.evaluator, prompt_version_id="v1", split="validation")
        self.assertIsNone(report.items[0].score)

    def test_complete_scores_retain_real_zero_and_the_expected_mean(self):
        report = self.load([0.0, 1.0])
        self.assertEqual(report.score, 0.5)
        self.assertEqual(report.scored_tasks, 2)
        empty = EvaluationReport("v1", "validation", self.root, ())
        self.assertIsNone(empty.score)
        self.assertIsNone(empty.total_tokens)

    def test_rejects_invalid_numeric_scores_including_booleans_and_non_finite_values(self):
        for value in (True, "0.5", -0.1, 1.1, math.nan, math.inf):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "fact_recall"):
                self.load([value])

    def test_comparison_does_not_accept_improvement_on_different_available_subsets(self):
        before = self.load([0.1, None])
        after = self.load([None, 1.0])
        decision = SelectionPolicy().compare(before, after)
        self.assertEqual(decision.status, "insufficient_data")
        self.assertFalse(decision.accepted)
        self.assertIsNone(decision.delta)
        self.assertIsNone(decision.regressions)
        self.assertIn("q0", decision.reason)
        self.assertIn("q1", decision.reason)

    def test_a_missing_measurement_on_either_side_blocks_quality_selection(self):
        complete = self.load([0.2, 0.2])
        incomplete = self.load([0.9, None])
        for before, after in ((complete, incomplete), (incomplete, complete), (incomplete, incomplete)):
            with self.subTest(before=before, after=after):
                decision = SelectionPolicy().compare(before, after)
                self.assertEqual(decision.status, "insufficient_data")
                self.assertIsNone(decision.delta)

    def test_complete_external_reports_can_still_be_compared(self):
        before = self.load([0.2, 0.5])
        after = self.load([0.6, 0.8])
        decision = SelectionPolicy().compare(before, after)
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.status, "compared")
        self.assertAlmostEqual(decision.delta, 0.35)
        self.assertEqual(decision.regressions, 0)

    def test_error_status_does_not_become_an_improving_baseline(self):
        before = self.load([0.0], ["error"])
        after = self.load([0.8])
        self.assertEqual(before.items[0].score, 0.0)
        self.assertIsNone(before.score)
        self.assertEqual(SelectionPolicy().compare(before, after).status, "insufficient_data")

    def test_comparison_requires_unique_task_ids_and_matching_splits(self):
        report = self.load([0.5])
        duplicate = replace(report, items=(report.items[0], report.items[0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            SelectionPolicy().compare(duplicate, report)
        with self.assertRaisesRegex(ValueError, "same evaluation split"):
            SelectionPolicy().compare(report, replace(report, split="training"))

    def test_reads_the_actual_shared_evaluator_fixture(self):
        examples = Path(__file__).resolve().parents[1] / "outputs" / "evaluation" / "example"
        report = load_evaluation_report(
            examples / "runner_results.jsonl", examples / "results.jsonl",
            prompt_version_id="v1", split="validation",
        )
        self.assertEqual([item.task_id for item in report.items], ["q001", "q002", "q003"])
        self.assertEqual([item.score for item in report.items], [1.0, 1.0, 1.0])
        self.assertEqual(report.scored_tasks, 3)
        self.assertEqual(report.score, 1.0)


if __name__ == "__main__":
    unittest.main()
