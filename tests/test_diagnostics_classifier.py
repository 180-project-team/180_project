"""Rule gating and HTTP response validation for diagnostic classifiers."""

import unittest
from unittest.mock import Mock, patch

from app.diagnostics import (
    ChatClassifier, Classification, DiagnosisStatus, FailureType, JevClassifier,
    diagnose_case, skim_case,
)


def ambiguity_case():
    return skim_case({
        "question": "Who directed both films?", "final_answer": "Alice", "stopped": "answer",
        "answer_f1": 0.5, "gold_ids": ["films"], "traceback": "private traceback",
        "trajectory": [
            {"action": "bm25_search", "observation": "One result", "hit_ids": ["films"]},
            {"action": "visit", "observation": "Alice directed the first. Bob directed the second.",
             "read_ids": ["films"], "raw_output": "private model output"},
            {"action": "answer", "observation": "(episode ended)"},
        ],
    })


class ClassifierTests(unittest.TestCase):
    def test_only_ambiguous_cases_are_sent_and_use_local_hypotheses(self):
        classifier = Mock()
        classifier.classify.return_value = Classification(FailureType.INCOMPLETE_ANSWER, 0.002)
        case = ambiguity_case()
        report = diagnose_case(case, classifier=classifier)
        self.assertEqual(report.failure_type, FailureType.INCOMPLETE_ANSWER)
        self.assertEqual(report.classifier_attempt.status, "accepted")
        self.assertEqual(report.classifier_attempt.cost_usd, 0.002)
        self.assertIn(["trajectory", 1, "read_ids"], [ref.path for ref in report.evidence_refs])
        self.assertEqual(report.hypotheses[0].hypothesis_id, "check_required_answer_facts")
        state, candidates = classifier.classify.call_args.args
        self.assertNotIn("private", str(state))
        self.assertIn("Bob", str(state))
        self.assertEqual(set(candidates), {
            FailureType.INCOMPLETE_ANSWER, FailureType.TOOL_OUTPUT_MISINTERPRETATION,
        })
        case["trajectory"][1]["read_ids"] = []
        report = diagnose_case(case, classifier=classifier)
        self.assertEqual(report.failure_type, FailureType.PREMATURE_STOP)
        classifier.classify.assert_called_once()

    def test_backend_failures_and_out_of_set_labels_preserve_local_report(self):
        case = ambiguity_case()
        baseline = diagnose_case(case)
        for result in (TimeoutError(), Classification(FailureType.CITATION_ERROR)):
            with self.subTest(result=result):
                classifier = Mock()
                if isinstance(result, Exception):
                    classifier.classify.side_effect = result
                else:
                    classifier.classify.return_value = result
                report = diagnose_case(case, classifier=classifier)
                self.assertEqual(report.status, DiagnosisStatus.FAILURE_FOUND)
                self.assertEqual(report.classifier_attempt.status, "unavailable")
                report.classifier_attempt = None
                self.assertEqual(report, baseline)

    @patch("app.diagnostics.classifier._post_json")
    def test_jev_choice_and_cost_are_parsed_and_threshold_can_abstain(self, post):
        post.return_value = {
            "answers": {"failure": {"type": "choice", "choice": "INCOMPLETE_ANSWER",
                                    "confidence": 0.75, "probabilities": {
                                        "INCOMPLETE_ANSWER": 0.75,
                                        "TOOL_OUTPUT_MISINTERPRETATION": 0.2, "UNKNOWN": 0.05,
                                    }}},
            "usage": {"cost": 0.0001},
        }
        report = diagnose_case(ambiguity_case(), classifier=JevClassifier("test"))
        self.assertEqual(report.failure_type, FailureType.UNKNOWN)
        self.assertEqual(report.classifier_attempt.status, "abstained")
        self.assertEqual(report.classifier_attempt.cost_usd, 0.0001)
        self.assertEqual(report.classifier_attempt.probabilities["INCOMPLETE_ANSWER"], 0.75)
        endpoint, _, payload, _ = post.call_args.args
        self.assertTrue(endpoint.endswith("/api/alpha/decisions"))
        self.assertEqual(set(payload["questions"]["failure"]["criteria"]), {
            "INCOMPLETE_ANSWER", "TOOL_OUTPUT_MISINTERPRETATION", "UNKNOWN",
        })

    @patch("app.diagnostics.classifier._post_json")
    def test_chat_requests_strict_schema_and_validates_output(self, post):
        post.return_value = {
            "choices": [{"message": {"content": '{"failure_type":"TOOL_OUTPUT_MISINTERPRETATION"}'}}],
        }
        report = diagnose_case(ambiguity_case(), classifier=ChatClassifier("https://example.com/v1", "test"))
        self.assertEqual(report.failure_type, FailureType.TOOL_OUTPUT_MISINTERPRETATION)
        endpoint, _, payload, _ = post.call_args.args
        self.assertEqual(endpoint, "https://example.com/v1/chat/completions")
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])
