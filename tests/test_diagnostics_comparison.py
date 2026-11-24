"""Verify comparison counts for the labeled diagnostic cases."""

import json
import unittest
from pathlib import Path
from unittest.mock import Mock

from examples.compare_skim_diagnostics import compare


class ComparisonTests(unittest.TestCase):
    def test_cases_without_supported_candidates_match_the_rule_based_baseline(self):
        fixture = Path(__file__).with_name("fixtures") / "skim_diagnostic_cases.jsonl"
        cases = [json.loads(line) for line in fixture.read_text(encoding="utf-8").splitlines()]
        classifier = Mock()
        scores = compare(cases, classifier)
        rules, hybrid = scores["rules"], scores["hybrid"]
        classifier.classify.assert_not_called()
        self.assertEqual(hybrid, rules)
        self.assertEqual(rules["cases"], 9)
        self.assertEqual(rules["correct_categories"], 7)
        self.assertEqual(hybrid["classifier_calls"], 0)
        self.assertEqual(hybrid["mean_cost_per_case_usd"], 0.0)
        self.assertAlmostEqual(hybrid["unknown_rate"], 7 / 9)
        self.assertEqual(hybrid["labeled_steps"], 2)
        self.assertEqual(hybrid["step_accuracy"], 1)
