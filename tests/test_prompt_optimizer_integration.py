"""Exercise the team's Runner, Evaluator and Diagnostics before the L1 call."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app.diagnostics import diagnose_run
from app.llm import LLMService
from app.prompt_optimizer import PromptOptimizer, load_evaluation_report
from app.prompt_optimizer.policy import SelectionPolicy
from app.runner import AgentRunner, CallableAdapter


class TeamPipelineIntegrationTests(unittest.TestCase):
    def test_consumes_team_artifacts_and_preserves_an_unavailable_metric(self):
        from app.evaluator import Evaluator

        facts = ["Alice directed the first film.", "Bob directed the second film."]
        answer = " ".join(facts)
        questions = [{"id": name, "question": f"Who directed both films in {name}?"}
                     for name in ("mixed", "metric_missing")]
        dataset = [{**row, "reference_answer": answer, "reference_facts": facts} for row in questions]

        class Agent:
            def ask(self, question):
                trajectory = [{"action": "visit", "observation": answer}]
                if "mixed" in question:
                    trajectory.insert(0, {"action": "bm25_search", "observation": "ERROR: Timeout"})
                return {"answer": facts[0], "trajectory": trajectory}

        before = 'SYSTEM_PROMPT = "Answer {question}."'
        after = 'SYSTEM_PROMPT = "Check the evidence and answer every part of {question}."'

        class Generator:
            def generate(self, agent_code, reports):
                report = reports[0]
                return {
                    "report_index": report["report_index"],
                    "hypothesis_id": report["hypotheses"][0]["hypothesis_id"],
                    "before": before, "after": after, "rationale": "Follow the supplied answer-coverage advice.",
                }

        with TemporaryDirectory() as directory:
            root = Path(directory)
            runner_path = root / "runner_results.jsonl"
            evaluator_dir = root / "evaluation"
            evaluator_dir.mkdir()
            # These are controlled metric responses, not a second implementation
            # of Evaluator or a claim about real agent quality.
            with patch("app.evaluator.evaluator.create_ragas_llm", return_value=object()), patch(
                "app.evaluator.evaluator.RagasMetrics", autospec=True,
            ) as metric_class:
                metric_class.return_value.calculate.side_effect = [
                    {"fact_recall": 0.4, "fact_precision": 1.0, "retrieval_fact_recall": 1.0, "faithfulness": 1.0},
                    {"fact_recall": None, "fact_precision": 1.0, "retrieval_fact_recall": 1.0, "faithfulness": 1.0},
                ]
                list(AgentRunner(lambda _: CallableAdapter(Agent)).run(questions, root, results_path=runner_path))
                Evaluator(LLMService(model="offline-integration-test")).evaluate(
                    evaluation_dataset=dataset, run_results_path=runner_path, output_dir=evaluator_dir,
                )
                self.assertEqual(metric_class.return_value.calculate.call_count, 2)

            evaluation_path = evaluator_dir / "results.jsonl"
            batch = diagnose_run(
                run_results_path=runner_path, evaluation_results_path=evaluation_path,
                output_dir=root / "diagnostics", run_id="run-1", version_id="v1",
            )
            diagnostic_path = root / "diagnostics" / "diagnoses.jsonl"
            inputs = [runner_path, evaluation_path, diagnostic_path]
            original = [path.read_bytes() for path in inputs]
            result = PromptOptimizer(generator=Generator()).optimize(diagnostic_path, root / "candidate", before)

            self.assertEqual(result.status, "proposed")
            self.assertEqual(result.candidate_code, after)
            self.assertEqual(batch.reports[0].suggested_scope, "STRUCTURAL")
            manifest = json.loads(result.result_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["diagnosis_ref"]["case_id"], "mixed")
            evaluation = load_evaluation_report(
                runner_path, evaluation_path, prompt_version_id="v1", split="validation",
            )
            self.assertEqual([item.status for item in evaluation.items], ["ok", "ok"])
            self.assertEqual([item.score for item in evaluation.items], [0.4, None])
            self.assertEqual(SelectionPolicy().compare(evaluation, evaluation).status, "insufficient_data")
            self.assertEqual([path.read_bytes() for path in inputs], original)


if __name__ == "__main__":
    unittest.main()
