"""Integration tests using Skim Search export structure."""

import unittest

from app.diagnostics import DiagnosisStatus, FailureType, SignalLevel, diagnose_case, normalize_trace, skim_case


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

    def test_invalid_visit_is_not_an_execution_error(self):
        report = diagnose_case(skim_case({
            "question": "Who directed it?", "final_answer": "Unknown", "stopped": "answer",
            "trajectory": [{"action": "visit", "args": {"rank": 9},
                            "observation": "ERROR: rank 9 out of range (last search had 2 results)."}],
        }))
        self.assertEqual(report.failure_type, FailureType.INVALID_TOOL_CALL)
        self.assertEqual(report.signals[0].component_id, "visit")

    def test_recovered_search_keeps_evidence_and_is_only_a_warning(self):
        search = {"action": "bm25_search", "args": {"query": "director"},
                  "query": "director"}
        report = diagnose_case(skim_case({
            "question": "Who directed it?", "final_answer": "Someone", "stopped": "answer",
            "trajectory": [dict(search, observation="ERROR: timed out"),
                           dict(search, observation="(2 matches)")],
        }))
        self.assertEqual(report.status, DiagnosisStatus.NO_FAILURE_DETECTED)
        recovered = next(signal for signal in report.signals if signal.rule_id == "tool_error")
        self.assertEqual(recovered.level, SignalLevel.WARNING)
        self.assertEqual([ref.path for ref in recovered.evidence_refs],
                         [["trajectory", 0, "observation"], ["trajectory", 1, "observation"]])

    def test_budget_stop_has_an_explicit_failure_category(self):
        report = diagnose_case(skim_case({
            "question": "Who directed it?", "final_answer": "", "stopped": "ctx_budget",
            "trajectory": [{"action": "budget", "observation": "Budget limit reached"}],
        }))
        self.assertEqual(report.failure_type, FailureType.PREMATURE_STOP)
        self.assertEqual(report.critical_step, 0)
