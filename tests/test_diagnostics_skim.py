"""Integration tests using Skim Search export structure."""

import unittest
from unittest.mock import Mock

from app.diagnostics import (
    DiagnosisStatus, EvidenceRef, FailureType, QualityThresholds, SignalLevel,
    diagnose, diagnose_case, normalize_trace, skim_case,
)


class SkimImportTests(unittest.TestCase):
    def test_scored_row_retains_event_links_and_case_metrics(self):
        row = {
            "instance_id": "2wiki-1", "question": "Who directed it?",
            "final_answer": "Unknown", "stopped": "answer",
            "answer_em": 0, "answer_f1": float("nan"),
            "gold_ids": ["annotation-id"],
            "trajectory": [
                {"action": "bm25_search", "args": {"query": "director"},
                 "query": "director", "observation": "ERROR: timed out",
                 "hit_ids": [], "read_ids": []},
                {"action": "answer", "observation": "(episode ended)"},
            ],
        }
        case = skim_case(row)
        self.assertEqual(case["metrics"], {"answer_em": 0, "answer_f1": None})
        self.assertNotIn("reference_document_ids", case)
        self.assertNotIn("gold_ids", case)
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

    def test_missed_or_unopened_document_leaves_the_cause_unknown(self):
        for hits in (["index-id"], ["annotation-id"]):
            with self.subTest(hits=hits):
                case = {
                    "question": "Who directed it?", "answer": "Unknown", "status": "ok",
                    "metrics": {"fact_recall": 0.5}, "trace_format": "skim_search",
                    "trajectory": [
                        {"action": "bm25_search", "observation": "Listing", "hit_ids": hits},
                        {"action": "answer", "observation": "(episode ended)"},
                    ],
                }
                report = diagnose_case(case)
                self.assertEqual(report, diagnose_case({**case, "reference_document_ids": ["annotation-id"]}))
                self.assertEqual(report.status, DiagnosisStatus.FAILURE_FOUND)
                self.assertEqual(report.failure_type, FailureType.UNKNOWN)
                self.assertIsNone(report.critical_step)
                self.assertEqual(report.hypotheses, [])
                self.assertEqual(report.evidence_refs, [EvidenceRef(["metrics", "fact_recall"])])

    def test_matching_read_ids_do_not_establish_context_coverage_or_trigger_fallback(self):
        case = {
            "question": "Who directed both films?", "answer": "Alice", "status": "ok",
            "metrics": {"fact_recall": 0.5, "retrieval_fact_recall": 0.5},
            "trace_format": "skim_search", "reference_document_ids": ["film"],
            "trajectory": [
                {"action": "bm25_search", "observation": "Listing", "hit_ids": ["film"]},
                {"action": "visit", "observation": "Document excerpt", "read_ids": ["film"]},
                {"action": "answer", "observation": "(episode ended)"},
            ],
        }
        classifier = Mock()
        report = diagnose_case(case, classifier=classifier)
        self.assertEqual(report.failure_type, FailureType.UNKNOWN)
        self.assertEqual([signal.rule_id for signal in report.signals],
                         ["low_quality_metrics", "possible_retrieval_failure"])
        self.assertEqual(report.hypotheses, [])
        self.assertIsNone(report.classifier_attempt)
        classifier.classify.assert_not_called()

    def test_partial_trace_keeps_document_data_and_local_evidence_resolvable(self):
        row = {
            "id": "skim-step-1", "status": "error", "error": "Backend unavailable",
            "trajectory": [
                None,
                {"action": "bm25_search", "observation": "ERROR: timeout", "hit_ids": ["skim-step-1"]},
                {"action": "visit", "read_ids": ["/trajectory/1"]},
            ],
        }
        events = normalize_trace(row)
        _, call, result, visit, execution_error = events
        report, = diagnose([row]).reports
        self.assertEqual(report.case_id, row["id"])
        self.assertEqual([event.step_index for event in events], [0, 1, 1, 2, None])
        self.assertEqual(call.tool_call_id, result.tool_call_id)
        self.assertNotEqual(call.tool_call_id, visit.tool_call_id)
        self.assertEqual([event.event_id for event in events],
                         ["/trajectory/0", "/trajectory/1", "/trajectory/1/observation", "/trajectory/2", "/error"])
        self.assertEqual([signal.rule_id for signal in report.signals], ["execution_error", "tool_error"])
        self.assertEqual(report.evidence_refs,
                         [EvidenceRef(["status"]), execution_error.payload_ref, result.error_ref])
        self.assertIsNone(report.critical_step)
        refs = [event.payload_ref for event in events] + report.evidence_refs
        values = []
        for ref in refs:
            value = row
            for part in ref.path:
                value = value[part]
            values.append(value)
        self.assertEqual(values[1]["hit_ids"], [report.case_id])
        self.assertEqual(values[3]["read_ids"], [call.event_id])
        self.assertEqual(values[-3:], ["error", "Backend unavailable", "ERROR: timeout"])

    def test_low_score_without_references_has_unknown_cause(self):
        case = skim_case({
            "question": "Who directed it?", "final_answer": "Unknown", "stopped": "answer",
            "answer_f1": 0.5, "trajectory": [{"action": "answer"}],
        })
        report = diagnose_case(case)
        self.assertEqual(report.status, DiagnosisStatus.FAILURE_FOUND)
        self.assertEqual(report.failure_type, FailureType.UNKNOWN)
        self.assertEqual(report.hypotheses, [])
        report = diagnose_case(case, quality_thresholds=QualityThresholds(answer_f1=0.4))
        self.assertEqual(report.status, DiagnosisStatus.NO_FAILURE_DETECTED)

    def test_final_context_metric_distinguishes_loss_from_omission(self):
        case = {
            "question": "Who directed it?", "answer": "Unknown", "status": "ok",
            "trajectory": [],
            "metrics": {"retrieval_fact_recall": 1, "fact_recall": 0,
                        "final_context_fact_recall": 0.5},
        }
        self.assertEqual(diagnose_case(case).failure_type, FailureType.CONTEXT_LOSS)
        case["metrics"]["final_context_fact_recall"] = 1
        self.assertEqual(diagnose_case(case).failure_type, FailureType.INCOMPLETE_ANSWER)
        del case["metrics"]["final_context_fact_recall"]
        self.assertEqual(diagnose_case(case).failure_type, FailureType.UNKNOWN)
