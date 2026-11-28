"""Interpret supplied Evaluator scores together with Runner observations."""

import unittest
from unittest.mock import Mock

from app.diagnostics import (
    Confidence, DiagnosisStatus, EvidenceRef, FailureType, QualityThresholds,
    SignalLevel, SuggestedScope, diagnose, diagnose_case,
)


class QualityDiagnosticsTests(unittest.TestCase):
    def check(self, **fields):
        row = {"question": "Where and when?", "answer": "Paris", "status": "ok", **fields}
        report, = diagnose([row]).reports
        for ref in report.evidence_refs + [ref for signal in report.signals for ref in signal.evidence_refs]:
            value = row
            for part in ref.path:
                value = value[part]
        return report

    def test_each_low_metric_is_a_quality_observation(self):
        for name, rule_id, hypothesis_id in (
            ("fact_recall", "low_fact_recall", "check_required_answer_facts"),
            ("fact_precision", "low_fact_precision", "check_answer_relevance"),
            ("retrieval_fact_recall", "possible_retrieval_failure", "inspect_supporting_document_recall"),
            ("faithfulness", "low_faithfulness", "check_claim_grounding"),
        ):
            with self.subTest(name=name):
                report = self.check(metrics={name: 0.0})
                self.assertEqual(report.status, DiagnosisStatus.FAILURE_FOUND)
                self.assertEqual(report.failure_type, FailureType.UNKNOWN)
                self.assertEqual(report.confidence, Confidence.LOW)
                self.assertEqual(report.suggested_scope, SuggestedScope.UNKNOWN)
                self.assertIsNone(report.critical_step)
                self.assertEqual([signal.rule_id for signal in report.signals], ["low_quality_metrics", rule_id])
                self.assertEqual(report.evidence_refs, [EvidenceRef(["metrics", name])])
                self.assertEqual([item.hypothesis_id for item in report.hypotheses], [hypothesis_id])

    def test_recall_combination_changes_hypotheses_without_proving_cause(self):
        low = self.check(metrics={"fact_recall": 0.4, "retrieval_fact_recall": 0.3})
        self.assertEqual([signal.rule_id for signal in low.signals], [
            "low_quality_metrics", "possible_retrieval_failure", "low_fact_recall",
        ])
        self.assertEqual(low.failure_type, FailureType.UNKNOWN)
        self.assertEqual(low.signals[1].evidence_refs, [
            EvidenceRef(["metrics", "retrieval_fact_recall"]), EvidenceRef(["metrics", "fact_recall"]),
        ])
        high = self.check(metrics={"fact_recall": 0.4, "retrieval_fact_recall": 0.9})
        self.assertEqual(high.failure_type, FailureType.UNKNOWN)
        self.assertEqual(high.signals[1].rule_id, "answer_coverage_gap")
        self.assertIn("The final model context has not been measured", high.signals[1].rationale)
        self.assertEqual([item.hypothesis_id for item in high.hypotheses], [
            "check_required_answer_facts", "check_evidence_interpretation", "inspect_context_selection",
        ])
        self.assertIn(EvidenceRef(["metrics", "retrieval_fact_recall"]), high.evidence_refs)

    def test_precision_combination_separates_relevance_and_grounding(self):
        for faithfulness, expected_rules, expected_hypotheses in (
            (0.95, ["supported_precision_gap"], ["check_answer_relevance"]),
            (0.25, ["low_faithfulness", "unsupported_precision_gap"],
             ["check_claim_grounding", "check_claim_correctness"]),
        ):
            with self.subTest(faithfulness=faithfulness):
                report = self.check(metrics={
                    "fact_recall": 0.95, "retrieval_fact_recall": 0.95,
                    "fact_precision": 0.4, "faithfulness": faithfulness,
                })
                self.assertEqual(report.failure_type, FailureType.UNKNOWN)
                self.assertEqual([signal.rule_id for signal in report.signals],
                                 ["low_quality_metrics", *expected_rules])
                self.assertEqual([item.hypothesis_id for item in report.hypotheses], expected_hypotheses)
                self.assertEqual(set(tuple(ref.path) for ref in report.evidence_refs), {
                    ("metrics", "fact_precision"), ("metrics", "faithfulness"),
                })

    def test_missing_and_null_are_unavailable_and_invalid_scores_are_flagged(self):
        for fields in ({}, {"metrics": {}}, {"metrics": {"fact_recall": None}}):
            with self.subTest(fields=fields):
                report = self.check(**fields)
                self.assertEqual(report.status, DiagnosisStatus.NO_FAILURE_DETECTED)
                self.assertIn("correctness is not established", report.rationale)
                self.assertEqual(report.evidence_refs, [])
        for value in (True, "0.5", float("nan"), float("inf"), -0.01, 1.01):
            with self.subTest(value=value):
                report = self.check(metrics={"fact_recall": value, "fact_precision": None})
                self.assertEqual(report.status, DiagnosisStatus.NO_FAILURE_DETECTED)
                self.assertEqual([signal.rule_id for signal in report.signals], ["invalid_quality_metrics"])
                self.assertEqual(report.evidence_refs, [EvidenceRef(["metrics", "fact_recall"])])

    def test_thresholds_are_inclusive_and_configurable(self):
        metrics = {"fact_recall": 0.8, "fact_precision": 0.8, "retrieval_fact_recall": 0.8, "faithfulness": 0.8}
        self.assertEqual(self.check(metrics=metrics).signals, [])
        row = {"status": "ok", "question": "Where?", "answer": "Paris", "trajectory": [], "metrics": metrics}
        report = diagnose_case(row, quality_thresholds=QualityThresholds(fact_precision=0.85))
        self.assertEqual([signal.rule_id for signal in report.signals], ["low_quality_metrics", "supported_precision_gap"])
        with self.assertRaisesRegex(ValueError, "fact_precision"):
            QualityThresholds(fact_precision=1.1)

    def test_evaluation_error_is_independent_of_runner_failure(self):
        report = self.check(evaluation_status="error", metrics={"fact_recall": 0.0})
        self.assertEqual(report.status, DiagnosisStatus.INSUFFICIENT_DATA)
        self.assertEqual([signal.rule_id for signal in report.signals], ["evaluation_error"])
        self.assertEqual(report.signals[0].level, SignalLevel.WARNING)
        self.assertEqual(report.evidence_refs, [EvidenceRef(["evaluation_status"])])
        failed = self.check(status="error", error="Timeout", evaluation_status="error", metrics={"fact_recall": 0.0})
        self.assertEqual(failed.status, DiagnosisStatus.FAILURE_FOUND)
        self.assertEqual([signal.rule_id for signal in failed.signals], ["execution_error", "evaluation_error"])

    def test_high_scores_preserve_the_observed_tool_error(self):
        report = self.check(evaluation_status="ok", metrics={
            "fact_recall": 1.0, "fact_precision": 1.0, "retrieval_fact_recall": 1.0, "faithfulness": 1.0,
        }, trajectory=[{"action": "bm25_search", "observation": "ERROR: Timeout"}])
        self.assertEqual(report.failure_type, FailureType.TOOL_EXECUTION_FAILURE)
        self.assertEqual([signal.rule_id for signal in report.signals], ["tool_error"])
        self.assertEqual(report.evidence_refs, [EvidenceRef(["trajectory", 0, "observation"])])
        self.assertIsNone(report.critical_step)

    def test_quality_hypotheses_without_tool_observations_do_not_trigger_classifier(self):
        classifier = Mock()
        report = diagnose_case({
            "question": "Where and when?", "answer": "Paris", "status": "ok", "trajectory": [],
            "metrics": {"fact_recall": 0.3, "retrieval_fact_recall": 0.9, "fact_precision": 0.2},
        }, classifier=classifier)
        self.assertGreater(len(report.hypotheses), 1)
        self.assertIsNone(report.classifier_attempt)
        classifier.classify.assert_not_called()
