"""Integration tests using Skim Search export structure."""

import unittest

from app.diagnostics import FailureType, diagnose_case, normalize_trace, skim_case


class SkimImportTests(unittest.TestCase):
    def test_scored_row_retains_event_links_and_case_metrics(self):
        row = {
            "instance_id": "2wiki-1", "question": "Who directed it?",
            "final_answer": "Unknown", "stopped": "answer",
            "answer_em": 0, "answer_f1": float("nan"),
            "trajectory": [
                {"action": "bm25_search", "args": {"query": "director"},
                 "query": "director", "observation": "ERROR: timed out",
                 "hit_ids": [], "read_ids": []},
                {"action": "answer", "observation": "(episode ended)"},
            ],
        }
        case = skim_case(row)
        self.assertEqual(case["metrics"], {"answer_em": 0, "answer_f1": None})
        call, result, *_ = normalize_trace(case)
        self.assertEqual(call.tool_call_id, result.tool_call_id)
        self.assertEqual(call.search_query, "director")
        self.assertEqual(result.error_ref.path, ["trajectory", 0, "observation"])
        report = diagnose_case(case)
        self.assertEqual(report.case_id, "2wiki-1")
        self.assertEqual(report.failure_type, FailureType.TOOL_EXECUTION_FAILURE)

    def test_skipped_rows_are_not_diagnosable(self):
        with self.assertRaisesRegex(ValueError, "Skipped episode"):
            skim_case({"instance_id": "2wiki-2", "skipped": "no_units"})
