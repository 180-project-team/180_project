"""Verify rule-based and hybrid comparison with a scripted LLM response."""

import json
import unittest
from pathlib import Path
from unittest.mock import create_autospec

from app.diagnostics import LLMClassifier
from app.llm import LLMService
from examples.compare_skim_diagnostics import compare


class ComparisonTests(unittest.TestCase):
    def test_hybrid_comparison_recovers_two_labeled_cases_with_a_scripted_service(self):
        fixture = Path(__file__).with_name("fixtures") / "skim_diagnostic_cases.jsonl"
        cases = [json.loads(line) for line in fixture.read_text(encoding="utf-8").splitlines()]
        service = create_autospec(LLMService, instance=True, spec_set=True)

        def respond(messages, **options):
            answer = json.loads(messages[1]["content"])["answer"]
            category = {"Alice": "INCOMPLETE_ANSWER", "Bob and Alice": "TOOL_OUTPUT_MISINTERPRETATION"}
            return json.dumps({"failure_type": category.get(answer, "UNKNOWN")})

        service.chat.side_effect = respond
        scores = compare(cases, LLMClassifier(service))
        rules, hybrid = scores["rules"], scores["hybrid"]
        self.assertEqual(rules["cases"], 9)
        self.assertEqual(rules["correct_categories"], 7)
        self.assertEqual(rules["classifier_calls"], 0)
        self.assertEqual(hybrid["correct_categories"], 9)
        self.assertEqual(hybrid["classifier_calls"], 5)
        self.assertEqual(service.chat.call_count, 5)
        self.assertIsNone(hybrid["mean_cost_per_case_usd"])
        self.assertAlmostEqual(hybrid["unknown_rate"], 5 / 9)
        self.assertEqual(hybrid["labeled_steps"], 2)
        self.assertEqual(hybrid["step_accuracy"], 1)
