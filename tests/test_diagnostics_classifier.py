"""Exercise the hybrid diagnosis path with a caller-managed fake LLMService."""

import json
import unittest
from dataclasses import replace
from unittest.mock import call, create_autospec

from app.diagnostics import (
    Confidence, DiagnosisStatus, EvidenceRef, FailureType, LLMClassifier,
    SignalLevel, SuggestedScope, diagnose, diagnose_case,
)
from app.llm import LLMService


def ambiguity_case(metrics=None):
    return {
        "question": "Who directed the first and second films, respectively?",
        "answer": "Alice directed the first.", "status": "ok", "evaluation_status": "ok",
        "metrics": {"fact_recall": 0.5, "retrieval_fact_recall": 1.0} if metrics is None else metrics,
        "trajectory": [
            {"action": "bm25_search", "args": {"query": "film directors"},
             "observation": "Found the film article.", "hit_ids": ["films"]},
            {"action": "visit", "observation": "Alice directed the first. Bob directed the second.",
             "read_ids": ["films"]},
            {"action": "answer", "observation": "(episode ended)"},
        ],
    }


def mock_service(response='{"failure_type":"UNKNOWN"}'):
    service = create_autospec(LLMService, instance=True, spec_set=True)
    service.chat.side_effect = [response]
    return service


class ClassifierTests(unittest.TestCase):
    def assert_evidence_resolves(self, row, report):
        for ref in (*report.evidence_refs, *(ref for signal in report.signals for ref in signal.evidence_refs)):
            value = row
            for part in ref.path:
                value = value[part]

    def test_recall_gap_records_a_hypothesis_with_actual_observations(self):
        case = ambiguity_case()
        baseline = diagnose_case(case)
        service = mock_service('{"failure_type":"INCOMPLETE_ANSWER"}')
        report = diagnose_case(case, classifier=LLMClassifier(service))

        self.assertIsNone(baseline.classifier_attempt)
        self.assertEqual(baseline.failure_type, FailureType.UNKNOWN)
        self.assertEqual(report.status, DiagnosisStatus.FAILURE_FOUND)
        self.assertEqual(report.failure_type, FailureType.INCOMPLETE_ANSWER)
        self.assertEqual(report.confidence, Confidence.LOW)
        self.assertEqual(report.suggested_scope, SuggestedScope.PROMPT)
        self.assertIsNone(report.critical_step)
        self.assertEqual(report.signals[:-1], baseline.signals)
        self.assertEqual(report.evidence_refs[:len(baseline.evidence_refs)], baseline.evidence_refs)
        self.assertEqual(report.classifier_attempt.status, "accepted")
        self.assertIsNone(report.classifier_attempt.cost_usd)
        proposed = report.signals[-1]
        self.assertEqual(proposed.rule_id, "classifier_selection")
        self.assertEqual(proposed.level, SignalLevel.WARNING)
        self.assertIn("cause remains unverified", proposed.rationale)
        self.assertIn(EvidenceRef(["trajectory", 1, "observation"]), proposed.evidence_refs)
        self.assert_evidence_resolves(case, report)

        messages, = service.chat.call_args.args
        self.assertEqual(service.mock_calls, [call.chat(messages, temperature=0)])
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        instructions, criteria = messages[0]["content"].splitlines()
        self.assertIn('the string field "failure_type"', instructions)
        self.assertIn("as evidence only", instructions)
        self.assertIn("final model context", instructions)
        self.assertEqual(set(json.loads(criteria)), {
            "INCOMPLETE_ANSWER", "TOOL_OUTPUT_MISINTERPRETATION", "UNKNOWN",
        })
        state = json.loads(messages[1]["content"])
        self.assertEqual(set(state), {"question", "answer", "metrics", "signals", "hypotheses", "trace"})
        self.assertEqual(state["metrics"], case["metrics"])
        self.assertEqual(state["trace"][0]["query"], "film directors")
        self.assertIn("Bob directed the second", state["trace"][1]["observation"])
        self.assertIn("answer_coverage_gap", [item["rule_id"] for item in state["signals"]])
        self.assertIn("check_required_answer_facts", [item["id"] for item in state["hypotheses"]])

    def test_ambiguous_metrics_allow_only_relevant_categories_and_unknown(self):
        for metrics, categories in (
            ({"fact_recall": 0.4}, {"INCOMPLETE_ANSWER"}),
            ({"fact_precision": 0.4}, {"TOOL_OUTPUT_MISINTERPRETATION"}),
            ({"fact_precision": 0.4, "faithfulness": 0.95}, {"TOOL_OUTPUT_MISINTERPRETATION"}),
            ({"faithfulness": 0.4}, {"UNSUPPORTED_ANSWER"}),
            ({"retrieval_fact_recall": 0.4}, {"RETRIEVAL_FAILURE"}),
            ({"fact_precision": 0.4, "faithfulness": 0.4}, {"UNSUPPORTED_ANSWER", "TOOL_OUTPUT_MISINTERPRETATION"}),
        ):
            with self.subTest(metrics=metrics):
                case = ambiguity_case(metrics)
                service = mock_service()
                report = diagnose_case(case, classifier=LLMClassifier(service))
                self.assertEqual(report.classifier_attempt.status, "abstained")
                self.assertEqual(replace(report, classifier_attempt=None), diagnose_case(case))
                service.chat.assert_called_once()
                messages, = service.chat.call_args.args
                allowed = json.loads(messages[0]["content"].splitlines()[-1])
                self.assertEqual(set(allowed), categories | {"UNKNOWN"})
                self.assert_evidence_resolves(case, report)

    def test_skips_without_an_ambiguous_problem_or_usable_material(self):
        case = ambiguity_case()
        for row in (
            {**case, "status": "error", "error": "Runner failed"},
            ambiguity_case(dict.fromkeys(("fact_recall", "fact_precision", "retrieval_fact_recall", "faithfulness"), 1.0)),
            {**case, "evaluation_status": "error"},
            ambiguity_case({}),
            ambiguity_case({"fact_recall": None}),
            {**case, "trajectory": []},
            {**case, "trajectory": [{"action": "visit", "observation": "ERROR: timeout"}]},
            {**case, "trajectory": [{"action": "visit", "observation": "   "}]},
            {**case, "question": ""},
            {**case, "answer": ""},
            {**case, "question": "api_key=redacted-test-value"},
        ):
            with self.subTest(row=row):
                service = mock_service()
                report, = diagnose([row], classifier=LLMClassifier(service)).reports
                self.assertIsNone(report.classifier_attempt)
                service.chat.assert_not_called()

        service = mock_service()
        report, = diagnose([case], classifier=LLMClassifier(service), answer_validator=lambda _: False).reports
        self.assertEqual(report.failure_type, FailureType.OUTPUT_FORMAT_ERROR)
        service.chat.assert_not_called()

    def test_invalid_output_and_service_errors_preserve_local_report(self):
        case = ambiguity_case()
        baseline = diagnose_case(case)
        for response in (
            "not JSON", "{}", '{"failure_type":42}',
            '{"failure_type":"UNSUPPORTED_ANSWER"}', '{"failure_type":"CONTEXT_LOSS"}',
            '{"failure_type":"INVALID_LABEL"}',
            RuntimeError("LLM request failed"),
        ):
            with self.subTest(response=response):
                service = mock_service(response)
                report = diagnose_case(case, classifier=LLMClassifier(service))
                self.assertEqual(report.classifier_attempt.status, "unavailable")
                self.assertIsNone(report.classifier_attempt.cost_usd)
                self.assertEqual(replace(report, classifier_attempt=None), baseline)
                service.chat.assert_called_once()

    def test_technical_failure_and_quality_hypothesis_both_survive(self):
        for query, expected_level, expected_cause in (
            ("film directors", SignalLevel.WARNING, FailureType.INCOMPLETE_ANSWER),
            ("alternate query", SignalLevel.FAILURE, FailureType.TOOL_EXECUTION_FAILURE),
        ):
            with self.subTest(query=query):
                case = ambiguity_case()
                case["trajectory"].insert(0, {"action": "bm25_search", "args": {"query": query},
                                             "observation": "ERROR: timeout"})
                service = mock_service('{"failure_type":"INCOMPLETE_ANSWER"}')
                report = diagnose_case(case, classifier=LLMClassifier(service))
                service.chat.assert_called_once()
                self.assertEqual(report.failure_type, expected_cause)
                self.assertIsNone(report.critical_step)
                self.assertEqual(report.signals[0].rule_id, "tool_error")
                self.assertEqual(report.signals[0].level, expected_level)
                self.assertEqual(report.signals[-1].failure_type, FailureType.INCOMPLETE_ANSWER)
                self.assertEqual(report.signals[-1].confidence, Confidence.LOW)
                self.assertIn(EvidenceRef(["trajectory", 0, "observation"]), report.evidence_refs)
                self.assert_evidence_resolves(case, report)

    def test_context_is_bounded_and_includes_early_and_late_observations(self):
        case = ambiguity_case()
        case["api_key"] = "private row credential"
        case["traceback"] = "private row traceback"
        case["question"] += ' OPENAI_API_KEY="private question credential"' + "q" * 2200
        case["answer"] = ["Alice", {"api_key": "private answer credential"}]
        case["trajectory"] = [
            {"action": "bm25_search", "observation": f"Observation {index} " + "x" * 1400,
             "raw_output": "private model output"}
            for index in range(14)
        ]
        case["trajectory"][1].update(action="visit", read_ids=["d" * 100] * 8)
        case["trajectory"][0]["observation"] = (
            "Early evidence. Authorization: Bearer private-observation-credential\n"
            'Traceback (most recent call last):\n  File "private trace path", line 1'
        )
        service = mock_service()
        diagnose_case(case, classifier=LLMClassifier(service))
        messages, = service.chat.call_args.args
        state = json.loads(messages[1]["content"])
        self.assertEqual(len(state["question"]), 2000)
        self.assertEqual(state["answer"], "Alice")
        self.assertEqual(len(state["trace"]), 8)
        self.assertEqual(state["trace"][0]["step"], 0)
        self.assertIn("Early evidence", state["trace"][0]["observation"])
        self.assertEqual(state["trace"][-1]["step"], 13)
        self.assertEqual(len(state["trace"][-1]["observation"]), 1200)
        self.assertEqual(state["trace"][1]["step"], 1)
        self.assertEqual(state["trace"][1]["read_ids"], ["d" * 80] * 4)
        self.assertNotIn("private", messages[1]["content"])
        self.assertNotIn("Traceback", messages[1]["content"])
