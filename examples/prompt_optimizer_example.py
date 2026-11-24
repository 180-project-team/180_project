"""Deterministic example: replace with your agent adapter and LLM in production."""

import argparse
import json
from inspect import signature
from pathlib import Path
from tempfile import mkdtemp

from app.prompt_optimizer import (
    PromptPatch, PromptState, SelectionPolicy, digest, load_evaluation_report, propose_candidate,
)
from app.prompt_optimizer.storage import write_json
from app.runner import AgentRunner, CallableAdapter, load_questions

class DemoAgent:
    def __init__(self, state):
        self.state = state

    def ask(self, question):
        # A deliberately weak target; only the prompt changes between runs.
        instruction = self.state.get("system.core").template
        if "verify" not in instruction.lower():
            return {"answer": "unknown", "trajectory": ["answered_without_verification"]}
        answers = {"2+2": "4", "1+1": "2", "3+3": "6", "4+4": "8"}
        return {"answer": answers.get(question, "unknown"), "trajectory": ["verified_answer"]}


def make_adapter(state, seed):
    return CallableAdapter(lambda: DemoAgent(state))

class DemoEvaluationHarness:
    """Write simulated evaluation results for the arithmetic demo."""

    def __init__(self, target_factory=make_adapter):
        self.target_factory = target_factory

    def evaluate(self, state, tasks, *, seed, split, runs_root):
        def adapter_factory(_run_dir=None):
            return self.target_factory(state, seed)

        if "run_dir" in signature(AgentRunner.run).parameters:
            runs_root.mkdir(parents=True, exist_ok=True)
            run_dir = Path(mkdtemp(prefix="demo-", dir=runs_root))
            results_path = run_dir / "results.jsonl"
            runner = AgentRunner(adapter_factory)
            rows = list(runner.run(tasks, run_dir))
            with results_path.open("w", encoding="utf-8") as file:
                for row in rows:
                    print(json.dumps(row), file=file)
        else:
            runner = AgentRunner(adapter_factory, runs_root=runs_root)
            result = runner.run(tasks, metadata={
                "prompt_version_id": state.prompt_version_id,
                "seed": seed,
                "split": split,
            })
            rows = result.load_results()
            results_path = result.results_path

        evaluation_path = results_path.parent / "demo_evaluation.jsonl"
        with evaluation_path.open("w", encoding="utf-8") as file:
            for row in rows:
                score = float(row["answer"] != "unknown") if row["status"] == "ok" else None
                evaluation = {"id": row["id"], "status": row["status"], "fact_recall": score}
                print(json.dumps(evaluation), file=file)

        return load_evaluation_report(
            results_path,
            evaluation_path,
            prompt_version_id=state.prompt_version_id,
            split=split,
        )


class DemoGenerator:
    def generate(self, state, lesson):
        component = state.get(lesson.component_id)
        return PromptPatch(
            component_id=lesson.component_id, operation="APPEND_RULE",
            before_hash=digest(component.template),
            payload="Verify the final answer against available information.",
            rule_id="verify_answer", rationale="Reduce incorrect answers by adding verification",
            evidence_refs=lesson.evidence_refs,
        )


def make_generator():
    return DemoGenerator()

def demo_diagnosis(run_id, case_id):
    return {
        "run_id": run_id,
        "case_id": case_id,
        "failure_type": "UNSUPPORTED_ANSWER",
        "suggested_scope": "PROMPT",
        "hypotheses": [{
            "hypothesis_id": "verify_answer",
            "scope": "PROMPT",
            "advice": "Verify the final answer against available information.",
        }],
        "evidence_refs": [{"path": ["trajectory", 0]}],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run one demo prompt optimization cycle")
    parser.add_argument("--output", type=Path, default=Path("runs/prompt_optimizer_demo"))
    args = parser.parse_args(argv)

    folder = Path(__file__).parent
    state = PromptState.from_dict(json.loads((folder / "prompt_optimizer_state.json").read_text(encoding="utf-8")))
    training = load_questions(folder / "prompt_optimizer_train.jsonl")
    validation = load_questions(folder / "prompt_optimizer_validation.jsonl")
    runs_root = args.output / "agent_runs"
    harness = DemoEvaluationHarness()

    training_report = harness.evaluate(state, training, seed=42, split="optimize", runs_root=runs_root)
    diagnosis = demo_diagnosis(training_report.run_dir.name, training_report.items[0].task_id)
    candidate, patch = propose_candidate(state, diagnosis, make_generator())
    before = harness.evaluate(state, validation, seed=42, split="validation", runs_root=runs_root)
    after = harness.evaluate(candidate, validation, seed=42, split="validation", runs_root=runs_root)
    decision = SelectionPolicy().compare(before, after)

    write_json(args.output / "candidate.json", {
        "prompt_state": candidate.to_dict(), "patch": patch.to_dict(), "decision": vars(decision),
    })
    print(f"Validation: {before.score:.3f} -> {after.score:.3f}; accepted: {decision.accepted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
