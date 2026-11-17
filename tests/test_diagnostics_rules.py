"""Deterministic diagnostics using synthetic runner results only."""

import copy
import json
import unittest
from dataclasses import asdict

from app.diagnostics import (
    Confidence, EvidenceRef, FailureType, SignalLevel, TraceEvent,
    TraceEventType, check_rules, normalize_trace,
)


class DiagnosticRuleTests(unittest.TestCase):
    def check(self, result, **kwargs):
        original = copy.deepcopy(result)
        signals = check_rules(result, normalize_trace(result), **kwargs)
        self.assertEqual(result, original)
        json.dumps([asdict(signal) for signal in signals])
        for signal in signals:
            self.assertTrue(signal.evidence_refs)
            for ref in signal.evidence_refs:
                value = result
                for part in ref.path:
                    value = value[part]
        return signals

    def test_nonempty_answer_without_trace_has_no_violation(self):
        for answer in ("arbitrary text", ["42"], {"value": 42}, 0, False):
            with self.subTest(answer=answer):
                self.assertEqual(self.check({"status": "ok", "answer": answer}), [])

    def test_partial_data_does_not_imply_failure(self):
        for result in ({}, {"id": "a"}, {"trajectory": [None]}, {"status": "running"}):
            with self.subTest(result=result):
                self.assertEqual(self.check(result), [])

    def test_execution_error_does_not_invent_component_or_step(self):
        for result in ({"status": "error"}, {"status": "error", "error": "Timeout", "answer": None}):
            with self.subTest(result=result):
                signals = self.check(result)
                self.assertEqual(len(signals), 1)
                self.assertEqual(signals[0].rule_id, "execution_error")
                self.assertEqual(signals[0].failure_type, FailureType.UNKNOWN)
                self.assertEqual(signals[0].level, SignalLevel.FAILURE)
                self.assertIsNone(signals[0].step_index)

    def test_missing_and_empty_answers_after_success(self):
        for result in ({"status": "ok"}, *(
            {"status": "ok", "answer": answer} for answer in (None, "", " \n", [], {})
        )):
            with self.subTest(result=result):
                signal, = self.check(result)
                self.assertEqual(signal.rule_id, "missing_answer")
                self.assertEqual(signal.failure_type, FailureType.OUTPUT_FORMAT_ERROR)
                self.assertEqual(signal.evidence_refs[0].path,
                                 ["answer"] if "answer" in result else ["status"])

    def test_explicit_answer_contract(self):
        validator = lambda answer: isinstance(answer, list) and all(isinstance(x, str) for x in answer)
        self.assertEqual(self.check({"status": "ok", "answer": ["42"]}, answer_validator=validator), [])
        signal, = self.check({"status": "ok", "answer": "text"}, answer_validator=validator)
        self.assertEqual(signal.rule_id, "invalid_answer")
        self.assertEqual(signal.evidence_refs[0].path, ["answer"])

    def test_validator_exception_is_not_a_false_agent_failure(self):
        def broken(answer):
            raise ValueError("broken validator")
        with self.assertRaisesRegex(ValueError, "broken validator"):
            self.check({"status": "ok", "answer": "text"}, answer_validator=broken)

    def test_tool_error_is_an_observation_even_if_agent_later_succeeds(self):
        result = {"status": "ok", "answer": "42", "trajectory": [
            {"action": "bm25_search", "observation": "ERROR: Timeout"},
            {"action": "answer"},
        ]}
        signal, = self.check(result)
        self.assertEqual(signal.rule_id, "tool_error")
        self.assertEqual(signal.step_index, 0)
        self.assertEqual(signal.evidence_refs[0].path, ["trajectory", 0, "observation"])

    def test_error_word_in_document_text_does_not_trigger(self):
        result = {"trajectory": [
            {"action": "visit", "observation": "The document explains the error."},
            {"action": "visit", "observation": ""},
        ]}
        self.assertEqual(self.check(result), [])

    def test_repeated_search_counts_each_call_once(self):
        steps = [
            {"action": "bm25_search", "query": query, "observation": "Listing"}
            for query in ("same", "other", "same")
        ]
        self.assertEqual(self.check({"trajectory": steps[:2]}), [])
        signal, = self.check({"trajectory": steps})
        self.assertEqual(signal.level, SignalLevel.WARNING)
        self.assertEqual(signal.confidence, Confidence.LOW)
        self.assertEqual(signal.failure_type, FailureType.INEFFICIENT_LOOP)
        self.assertEqual(signal.step_index, 2)
        self.assertEqual([ref.path for ref in signal.evidence_refs],
                         [["trajectory", 0], ["trajectory", 2]])

    def test_search_query_matching_is_exact(self):
        result = {"trajectory": [{"action": "bm25_search", "query": query}
                                 for query in ("Q", "q", " ", " ")]}
        self.assertEqual(self.check(result), [])

    def test_rules_accept_events_from_an_independent_normalizer(self):
        result = {"custom": [{"failure": "timeout"}]}
        event = TraceEvent("custom-event", 4, TraceEventType.TOOL_RESULT,
                           "custom-tool", EvidenceRef(["custom", 0]),
                           error_ref=EvidenceRef(["custom", 0, "failure"]))
        signal, = check_rules(result, [event])
        self.assertEqual(signal.step_index, 4)
        self.assertEqual(signal.evidence_refs[0].path, ["custom", 0, "failure"])


if __name__ == "__main__":
    unittest.main()
