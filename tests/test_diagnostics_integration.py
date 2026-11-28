"""Exercise Runner, Evaluator and Diagnostics with offline agent and model responses."""

import copy
import json
import os
import unittest
from contextlib import chdir
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import create_autospec, patch

from app.diagnostics import (
    DiagnosisBatchResult, DiagnosisStatus, FailureType, SignalLevel, diagnose, diagnose_run, load_diagnostic_cases,
)
from app.llm import LLMService
from app.runner import AgentRunner, CallableAdapter
from examples.diagnose_runner import run_example
from examples.orchestrator_demo import ExampleOrchestrator


_FACTS = ["Alice directed the first film.", "Bob directed the second film."]


def _judge_responses(*, incomplete=False, unavailable=False):
    """Script model outputs for the four real RAGAS metrics in Evaluator order."""
    answer_facts = _FACTS[:1] if incomplete else _FACTS

    def statements(facts, verdicts):
        return {"statements": [
            {"statement": fact, "verdict": verdict, "reason": "Fixture judgment"}
            for fact, verdict in zip(facts, verdicts, strict=True)
        ]}

    yield "ClaimDecompositionOutput", {"claims": _FACTS}
    yield "NLIStatementOutput", statements(_FACTS, [1, 0 if incomplete else 1])
    yield "ClaimDecompositionOutput", {"claims": answer_facts}
    yield "NLIStatementOutput", statements(answer_facts, [1] * len(answer_facts))
    yield "ContextRecallOutput", {"classifications": [
        {"statement": fact, "attributed": 1, "reason": "Fixture observation"} for fact in _FACTS
    ]}
    if unavailable:
        yield "StatementGeneratorOutput", RuntimeError("Synthetic evaluator model failure")
    else:
        yield "StatementGeneratorOutput", {"statements": answer_facts}
        yield "NLIStatementOutput", statements(answer_facts, [1] * len(answer_facts))


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class DiagnosticsIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.run_dir = self.enterContext(TemporaryDirectory())

    def test_example_runs_all_scenarios_through_existing_runner(self):
        rows, batch = run_example(run_dir=self.run_dir)
        self.assertEqual(batch.summary.run_id, "demo-run")
        self.assertEqual(batch.summary.version_id, "demo-v1")
        self.assertEqual(sum(row["status"] == "error" for row in rows), 1)
        self.assertEqual([report.case_id for report in batch.reports],
                         [row["id"] for row in rows])
        self.assertEqual([report.status for report in batch.reports], [
            DiagnosisStatus.NO_FAILURE_DETECTED,
            DiagnosisStatus.FAILURE_FOUND,
            DiagnosisStatus.FAILURE_FOUND,
            DiagnosisStatus.FAILURE_FOUND,
            DiagnosisStatus.NO_FAILURE_DETECTED,
        ])
        self.assertEqual(batch.summary.analyzed_cases, 5)
        self.assertEqual(batch.summary.failed_cases, 3)
        self.assertEqual(batch.summary.failure_counts, {
            FailureType.OUTPUT_FORMAT_ERROR: 1,
            FailureType.UNKNOWN: 1,
            FailureType.TOOL_EXECUTION_FAILURE: 1,
        })
        tool_report = batch.reports[3]
        self.assertIsNone(tool_report.critical_step)
        self.assertEqual(tool_report.signals[0].step_index, 0)
        self.assertEqual(batch.reports[4].signals[0].level, SignalLevel.WARNING)
        original = copy.deepcopy(rows)
        diagnose(rows)
        self.assertEqual(rows, original)
        for row, report in zip(rows, batch.reports):
            for ref in report.evidence_refs:
                value = row
                for part in ref.path:
                    value = value[part]
        json.dumps(asdict(batch))

    def test_diagnosis_handles_runner_error_without_trajectory(self):
        class ErrorAgent:
            def ask(self, question):
                return {"error": "Synthetic failure", "trajectory": [
                    {"action": "bm25_search", "observation": "ERROR: Timeout"}
                ], "traceback": "Synthetic traceback"}
        runner = AgentRunner(lambda _: CallableAdapter(ErrorAgent))
        row, = runner.run([{"question": "question without dataset identifier"}], self.run_dir)
        self.assertEqual(row["id"], "1")
        self.assertEqual(row["status"], "error")
        self.assertIsNone(row["answer"])
        self.assertNotIn("trajectory", row)
        self.assertNotIn("traceback", row)
        report, = diagnose([row]).reports
        self.assertEqual(report.case_id, row["id"])
        self.assertEqual(report.failure_type, FailureType.UNKNOWN)
        self.assertIsNone(report.critical_step)
        self.assertEqual([signal.rule_id for signal in report.signals], ["execution_error"])

    def test_runner_evaluator_and_diagnostics_publish_consistent_reports(self):
        self.enterContext(patch.dict(os.environ, {"RAGAS_DO_NOT_TRACK": "true"}))
        from ragas import _analytics
        from ragas.llms import InstructorLLM

        from app.evaluator import Evaluator

        root = Path(self.run_dir).resolve()
        self.enterContext(patch.object(_analytics, "user_data_dir", return_value=str(root / "ragas")))
        for read_setting in (_analytics.do_not_track, _analytics.get_userid):
            read_setting.cache_clear()
            self.addCleanup(read_setting.cache_clear)
        evaluator_dir = root / "evaluation"
        caller_dir = root / "caller"
        evaluator_dir.mkdir()
        caller_dir.mkdir()
        runner_path = root / "runner_results.jsonl"
        evaluation_path = evaluator_dir / "results.jsonl"
        ids = ["good", "retried", "agent_error", "incomplete", "unscored", "metric_error"]
        answer = " ".join(_FACTS)
        questions = [{"id": case_id, "question": f"Who directed each film in case {case_id}?"} for case_id in ids]
        dataset = [{**row, "reference_answer": answer, "reference_facts": _FACTS} for row in questions]
        search = {"action": "bm25_search", "args": {"query": "film directors"}, "observation": "Film records found."}
        visit = {"action": "visit", "observation": answer}
        retry_trace = [{**search, "observation": "ERROR: Synthetic search timeout"}, search, visit]
        outcomes = {
            "good": {"answer": answer, "trajectory": [search, visit]},
            "retried": {"answer": answer, "trajectory": retry_trace},
            "agent_error": RuntimeError("Synthetic agent failure"),
            "incomplete": {"answer": _FACTS[0], "trajectory": retry_trace},
            "unscored": {"answer": answer},
            "metric_error": {"answer": answer, "trajectory": [search, visit]},
        }
        outcomes_by_question = {row["question"]: outcomes[row["id"]] for row in questions}

        class PipelineAgent:
            def ask(self, question):
                outcome = outcomes_by_question[question]
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome

        judge_responses = iter([
            *_judge_responses(), *_judge_responses(),
            *_judge_responses(incomplete=True), *_judge_responses(unavailable=True),
        ])

        async def generate_judgment(_model, prompt, response_model):
            name, response = next(judge_responses)
            self.assertEqual(response_model.__name__, name)
            self.assertNotIn("Synthetic search timeout", prompt)
            if isinstance(response, Exception):
                raise response
            return response_model.model_validate(response)

        service = create_autospec(LLMService, instance=True, spec_set=True)
        service.chat.return_value = '{"failure_type":"INCOMPLETE_ANSWER"}'
        runner = AgentRunner(lambda _: CallableAdapter(PipelineAgent))
        evaluator = Evaluator(LLMService(model="offline-integration-test"))
        with chdir(caller_dir):
            ExampleOrchestrator(runner).execute(questions, root, results_path=runner_path)
            original_runner = runner_path.read_bytes()
            with patch.object(InstructorLLM, "agenerate", autospec=True, side_effect=generate_judgment) as judge:
                with self.assertLogs("app.evaluator.metrics", level="ERROR") as logs:
                    evaluator.evaluate(evaluation_dataset=dataset, run_results_path=runner_path, output_dir=evaluator_dir)
            original_evaluation = evaluation_path.read_bytes()
            local = diagnose_run(run_results_path=runner_path, evaluation_results_path=evaluation_path,
                                 output_dir=root / "local", run_id="test-run", version_id="test-version")
            hybrid = diagnose_run(run_results_path=runner_path, evaluation_results_path=evaluation_path,
                                  output_dir=root / "hybrid", run_id="test-run", version_id="test-version",
                                  llm_service=service)

        self.assertEqual(judge.await_count, 27)
        self.assertIsNone(next(judge_responses, None))
        self.assertEqual(len(logs.records), 1)
        self.assertIn("faithfulness", logs.records[0].getMessage())
        self.assertEqual({path.name for path in evaluator_dir.iterdir()}, {
            "results.jsonl", "results.csv", "summary.json", "summary.csv",
        })
        evaluation_summary = json.loads((evaluator_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(evaluation_summary["faithfulness"], {"mean": 1.0, "count": 3})
        self.assertEqual(runner_path.read_bytes(), original_runner)
        self.assertEqual(evaluation_path.read_bytes(), original_evaluation)

        runner_rows = _read_jsonl(runner_path)
        evaluation_rows = _read_jsonl(evaluation_path)
        cases = load_diagnostic_cases(runner_path, evaluation_path)
        self.assertEqual([row["id"] for row in runner_rows], ids)
        self.assertEqual([row["id"] for row in evaluation_rows], ids)
        self.assertEqual([case["case_id"] for case in cases], ids)
        self.assertEqual([(case["status"], case["evaluation_status"]) for case in cases], [
            ("ok", "ok"), ("ok", "ok"), ("error", "error"), ("ok", "ok"), ("ok", "error"), ("ok", "ok"),
        ])
        metric_names = ("fact_recall", "fact_precision", "retrieval_fact_recall", "faithfulness")
        self.assertEqual([tuple(case["metrics"][name] for name in metric_names) for case in cases], [
            (1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1.0), (None, None, None, None),
            (0.5, 1.0, 1.0, 1.0), (None, None, None, None), (1.0, 1.0, 1.0, None),
        ])
        self.assertEqual(evaluation_rows[1]["failed_tool_calls_count"], 1)
        self.assertEqual(evaluation_rows[1]["successful_tool_calls_count"], 2)

        for mode, batch in (("local", local), ("hybrid", hybrid)):
            self.assertIsInstance(batch, DiagnosisBatchResult)
            directory = root / mode
            self.assertEqual({path.name for path in directory.iterdir()}, {"diagnoses.jsonl", "summary.json"})
            reports = _read_jsonl(directory / "diagnoses.jsonl")
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual([report["case_id"] for report in reports], ids)
            self.assertEqual(reports, [asdict(report) for report in batch.reports])
            self.assertEqual(summary, asdict(batch.summary))
            self.assertEqual((summary["analyzed_cases"], summary["failed_cases"]), (len(ids), 2))
            self.assertEqual(summary["report_indices"], list(range(len(ids))))
            self.assertEqual((summary["run_id"], summary["version_id"]), ("test-run", "test-version"))
            tool_pattern, = [pattern for pattern in summary["patterns"] if pattern["rule_id"] == "tool_error"]
            self.assertEqual(tool_pattern["report_indices"], [1, 3])
            for joined, runner_row, evaluation_row, report in zip(cases, runner_rows, evaluation_rows, reports, strict=True):
                self.assertEqual((report["run_id"], report["version_id"]), ("test-run", "test-version"))
                signal_refs = [ref for signal in report["signals"] for ref in signal["evidence_refs"]]
                for ref in [*report["evidence_refs"], *signal_refs]:
                    field, *parts = ref["path"]
                    value = joined[field]
                    if field == "metrics":
                        expected = evaluation_row[parts[0]]
                    elif field == "evaluation_status":
                        expected = evaluation_row["status"]
                    else:
                        expected = runner_row[field]
                        for part in parts:
                            expected = expected[part]
                    for part in parts:
                        value = value[part]
                    self.assertEqual(value, expected)

        by_id = {report.case_id: report for report in hybrid.reports}
        self.assertEqual([report.status for report in hybrid.reports], [
            DiagnosisStatus.NO_FAILURE_DETECTED, DiagnosisStatus.NO_FAILURE_DETECTED, DiagnosisStatus.FAILURE_FOUND,
            DiagnosisStatus.FAILURE_FOUND, DiagnosisStatus.INSUFFICIENT_DATA, DiagnosisStatus.NO_FAILURE_DETECTED,
        ])
        self.assertEqual([signal.rule_id for signal in by_id["agent_error"].signals], ["execution_error", "evaluation_error"])
        self.assertEqual([signal.rule_id for signal in by_id["unscored"].signals], ["evaluation_error"])
        self.assertEqual(by_id["retried"].signals[0].level, SignalLevel.WARNING)
        incomplete = by_id["incomplete"]
        self.assertEqual(incomplete.failure_type, FailureType.INCOMPLETE_ANSWER)
        self.assertEqual(incomplete.signals[0].rule_id, "tool_error")
        self.assertIn("effect of the error on the final answer is unknown", incomplete.signals[0].rationale)
        self.assertEqual(incomplete.classifier_attempt.status, "accepted")
        self.assertEqual([report.case_id for report in hybrid.reports if report.classifier_attempt], ["incomplete"])
        self.assertTrue(all(report.classifier_attempt is None for report in local.reports))
        self.assertEqual(local.reports[3].failure_type, FailureType.UNKNOWN)
        self.assertEqual(local.summary.failure_counts, {FailureType.UNKNOWN: 2})
        self.assertEqual(hybrid.summary.failure_counts, {FailureType.UNKNOWN: 1, FailureType.INCOMPLETE_ANSWER: 1})
        service.chat.assert_called_once()


if __name__ == "__main__":
    unittest.main()
