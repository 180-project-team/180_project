"""Exercise comparison counters with a deterministic substitute for the API."""

import json
import unittest
from pathlib import Path

from app.diagnostics import Classification, FailureType
from examples.compare_skim_diagnostics import compare


class FixtureClassifier:
    def classify(self, state, candidates):
        category = {
            "Alice": FailureType.INCOMPLETE_ANSWER,
            "Bob and Alice": FailureType.TOOL_OUTPUT_MISINTERPRETATION,
        }[state["answer"]]
        if category not in candidates:
            raise ValueError("Expected category was not proposed by the rules")
        return Classification(category, cost_usd=0.01)


class ComparisonTests(unittest.TestCase):
    def test_only_two_labeled_cases_need_a_classifier(self):
        fixture = Path(__file__).with_name("fixtures") / "skim_diagnostic_cases.jsonl"
        cases = [json.loads(line) for line in fixture.read_text(encoding="utf-8").splitlines()]
        scores = compare(cases, FixtureClassifier())
        rules, hybrid = scores["rules"], scores["hybrid"]
        self.assertEqual(rules["cases"], 9)
        self.assertEqual(rules["correct_categories"], 7)
        self.assertEqual(hybrid["correct_categories"], 9)
        self.assertEqual(hybrid["classifier_calls"], 2)
        self.assertAlmostEqual(hybrid["mean_cost_per_case_usd"], 0.02 / 9)
        self.assertAlmostEqual(hybrid["unknown_rate"], 2 / 9)
        self.assertEqual(hybrid["labeled_steps"], 3)
        self.assertEqual(hybrid["step_accuracy"], 1)
