"""Integration tests using Runner rows with Skim Search trajectories."""

import unittest
from unittest.mock import Mock

from app.diagnostics import (
    DiagnosisStatus, EvidenceRef, FailureType, QualityThresholds, SignalLevel,
    diagnose, diagnose_case, normalize_trace,
)


class SkimTrajectoryTests(unittest.TestCase):
    def test_scored_row_retains_event_links_and_case_metrics(self):
        case = {
            "id": "2wiki-1", "question": "Who directed it?",
            "answer": "Unknown", "status": "ok", "evaluation_status": "ok",
            "metrics": {"fact_recall": 0, "fact_precision": None},
            "trajectory": [
                {"action": "bm25_search", "args": {"query": "director"},
                 "query": "director", "observation": "ERROR: timed out",
                 "hit_ids": [], "read_ids": []},
                {"action": "answer", "observation": "(episode ended)"},
            ],
        }
        call, result, *_ = normalize_trace(case)
        self.assertEqual(call.tool_call_id, result.tool_call_id)
        self.assertEqual(call.search_query, "director")
        self.assertEqual(result.error_ref.path, ["trajectory", 0, "observation"])
        report = diagnose_case(case)
        self.assertEqual(report.case_id, "2wiki-1")
        self.assertEqual(report.failure_type, FailureType.TOOL_EXECUTION_FAILURE)
        self.assertIn(EvidenceRef(["metrics", "fact_recall"]), report.evidence_refs)

    def test_invalid_visit_is_not_an_execution_error(self):
        report = diagnose_case({
            "question": "Who directed it?", "answer": "Unknown", "status": "ok", "metrics": {},
            "trajectory": [{"action": "visit", "args": {"rank": 9},
                            "observation": "ERROR: rank 9 out of range (last search had 2 results)."}],
        })
        self.assertEqual(report.failure_type, FailureType.INVALID_TOOL_CALL)
        self.assertEqual(report.signals[0].component_id, "visit")

    def test_recovered_search_keeps_evidence_and_is_only_a_warning(self):
        search = {"action": "bm25_search", "args": {"query": "director"},
                  "query": "director"}
        report = diagnose_case({
            "question": "Who directed it?", "answer": "Someone", "status": "ok", "metrics": {},
            "trajectory": [dict(search, observation="ERROR: timed out"),
                           dict(search, observation="(2 matches)")],
        })
        self.assertEqual(report.status, DiagnosisStatus.NO_FAILURE_DETECTED)
        recovered = next(signal for signal in report.signals if signal.rule_id == "tool_error")
        self.assertEqual(recovered.level, SignalLevel.WARNING)
        self.assertEqual([ref.path for ref in recovered.evidence_refs],
                         [["trajectory", 0, "observation"], ["trajectory", 1, "observation"]])

    def test_budget_stop_has_an_explicit_failure_category(self):
        report = diagnose_case({
            "question": "Who directed it?", "answer": "", "status": "unknown", "stopped": "ctx_budget",
            "metrics": {},
            "trajectory": [{"action": "budget", "observation": "Budget limit reached"}],
        })
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
                self.assertEqual([hypothesis.hypothesis_id for hypothesis in report.hypotheses],
                                 ["check_required_answer_facts"])
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
                         ["low_quality_metrics", "possible_retrieval_failure", "low_fact_recall"])
        self.assertEqual([hypothesis.hypothesis_id for hypothesis in report.hypotheses],
                         ["inspect_supporting_document_recall", "check_required_answer_facts"])
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
        case = {
            "question": "Who directed it?", "answer": "Unknown", "status": "ok",
            "metrics": {"fact_recall": 0.5}, "trajectory": [{"action": "answer"}],
        }
        report = diagnose_case(case)
        self.assertEqual(report.status, DiagnosisStatus.FAILURE_FOUND)
        self.assertEqual(report.failure_type, FailureType.UNKNOWN)
        self.assertEqual([hypothesis.hypothesis_id for hypothesis in report.hypotheses],
                         ["check_required_answer_facts"])
        report = diagnose_case(case, quality_thresholds=QualityThresholds(fact_recall=0.4))
        self.assertEqual(report.status, DiagnosisStatus.NO_FAILURE_DETECTED)
