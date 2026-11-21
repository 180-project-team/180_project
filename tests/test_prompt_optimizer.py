"""Prompt optimizer unit and end-to-end tests; no external model required."""

import json
import tempfile
import unittest
from pathlib import Path

from app.prompt_optimizer import (
    BehaviorAnalyzer, ExperimentStore, LLMPatchGenerator, OptimizationConfig,
    PromptComponent, PromptGuard, PromptOptimizer, PromptPatch, PromptRenderer,
    PromptState, RunnerEvaluationHarness, SelectionConfig, SelectionPolicy, digest,
)
from app.prompt_optimizer.evaluation import EvaluationItem, EvaluationReport


class PromptModelTests(unittest.TestCase):
    def setUp(self):
        self.state = PromptState(components=(
            PromptComponent("system.core", "system", "Answer {question}", ("question",)),
            PromptComponent("policy.safety", "system", "Keep data private", editable=False),
        ))

    def patch(self, component_id="system.core", operation="REPLACE_COMPONENT", payload="Answer carefully {question}", before_hash=None):
        return PromptPatch(
            component_id, operation,
            before_hash or digest(self.state.get(component_id).template),
            payload, "Improve evidence handling", ("run/results.jsonl#1",),
        )

    def test_state_is_content_addressed_and_round_trips(self):
        loaded = PromptState.from_dict(self.state.to_dict())
        self.assertEqual(loaded, self.state)
        self.assertEqual(loaded.prompt_version_id, self.state.prompt_version_id)
        with self.assertRaises(ValueError):
            PromptState.from_dict({**self.state.to_dict(), "prompt_version_id": "tampered"})
        with self.assertRaises(ValueError):
            PromptState((self.state.components[0], self.state.components[0]))

    def test_guard_does_not_mutate_parent(self):
        candidate = PromptGuard().apply(self.state, self.patch())
        self.assertEqual(candidate.parent_id, self.state.prompt_version_id)
        self.assertNotEqual(candidate.prompt_version_id, self.state.prompt_version_id)
        self.assertEqual(self.state.get("system.core").template, "Answer {question}")
        self.assertEqual(candidate.get("policy.safety").template, "Keep data private")

    def test_guard_rejects_fixed_stale_and_bad_variables(self):
        with self.assertRaisesRegex(ValueError, "Fixed"):
            PromptGuard().apply(self.state, self.patch("policy.safety"))
        with self.assertRaisesRegex(ValueError, "Stale"):
            PromptGuard().apply(self.state, self.patch(before_hash="bad"))
        with self.assertRaisesRegex(ValueError, "placeholders"):
            PromptGuard().apply(self.state, self.patch(payload="No placeholder"))
        with self.assertRaisesRegex(ValueError, "growth"):
            PromptGuard(max_growth_chars=0).apply(self.state, self.patch())

    def test_guard_rule_mutations_and_secret_check(self):
        guard = PromptGuard()
        add = PromptPatch("system.core", "APPEND_RULE", digest(self.state.get("system.core").template),
                          "Check your answer.", "Lesson", ("trace#1",), rule_id="verify")
        updated = guard.apply(self.state, add)
        self.assertIn("- [verify] Check your answer.", updated.get("system.core").template)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            guard.apply(updated, PromptPatch("system.core", "APPEND_RULE", digest(updated.get("system.core").template),
                                             "Check again", "Lesson", ("trace#1",), rule_id="verify"))
        replace = PromptPatch("system.core", "UPDATE_RULE", digest(updated.get("system.core").template),
                              "Use sources", "Lesson", ("trace#1",), rule_id="verify")
        changed = guard.apply(updated, replace)
        self.assertIn("Use sources", changed.get("system.core").template)
        removed = guard.apply(changed, PromptPatch("system.core", "REMOVE_RULE", digest(changed.get("system.core").template),
                                                  "", "Lesson", ("trace#1",), rule_id="verify"))
        self.assertEqual(removed.get("system.core").template.strip(), self.state.get("system.core").template)
        with self.assertRaisesRegex(ValueError, "secret"):
            guard.apply(self.state, self.patch(payload="Never reveal github_pat_" + "A" * 30 + " {question}"))

    def test_renderer_is_deterministic_and_checks_variables(self):
        renderer = PromptRenderer()
        first = renderer.render(self.state, {"question": "where?"})
        second = renderer.render(self.state, {"question": "where?"})
        self.assertEqual(first.content_hash, second.content_hash)
        self.assertEqual(first.messages[0]["content"], "Answer where?")
        with self.assertRaisesRegex(ValueError, "Missing"):
            renderer.render(self.state)
        malformed = PromptState((PromptComponent("system.core", "system", "Hello {name}"),))
        with self.assertRaisesRegex(ValueError, "variables"):
            renderer.validate(malformed)

    def test_llm_generator_parses_structured_patch(self):
        class FakeLLM:
            def chat(self, messages, **kwargs):
                return json.dumps({"operation": "APPEND_RULE", "payload": "Check sources", "rule_id": "check", "rationale": "Evidence"})
        from app.prompt_optimizer.diagnosis import PromptLesson
        lesson = PromptLesson("incorrect_answer-1", "system.core", "Check evidence", ("run#1",))
        patch = LLMPatchGenerator(FakeLLM()).generate(self.state, lesson)
        self.assertEqual(patch.operation, "APPEND_RULE")
        self.assertEqual(patch.component_id, "system.core")
        self.assertEqual(patch.before_hash, digest("Answer {question}"))


class DiagnosisTests(unittest.TestCase):
    def test_uses_execution_trajectory_for_pattern_and_attribution(self):
        from app.prompt_optimizer.diagnosis import PromptAttribution, LessonExtractor
        state = PromptState((
            PromptComponent("verifier.instructions", "system", "Check your answer."),
            PromptComponent("output.format", "system", "Return JSON."),
        ))
        report = EvaluationReport("v1", "optimize", Path("runs/sample"), (
            EvaluationItem("q-1", 0.0, "ok", 2.0, None, "runs/sample/results.jsonl#q-1",
                           trace_events=({"event": "skipped_verification"},)),
        ))
        pattern = BehaviorAnalyzer().analyze(report)[0]
        self.assertEqual(pattern.kind, "insufficient_verification")
        self.assertEqual(pattern.affected_tasks, ("q-1",))
        diagnosis = PromptAttribution().attribute(pattern, state)
        self.assertEqual(diagnosis.component_id, "verifier.instructions")
        lesson = LessonExtractor().extract(pattern, diagnosis)
        self.assertIn("verify", lesson.instruction)
        self.assertEqual(lesson.evidence_refs, pattern.evidence_refs)


class ComparatorTests(unittest.TestCase):
    def report(self, scores, tokens=None):
        items = tuple(EvaluationItem(str(index), score, "ok", 0.0,
                                     tokens[index] if tokens else None, "r#1")
                      for index, score in enumerate(scores))
        return EvaluationReport("id", "validation", Path("dummy"), items)

    def test_improvement_and_guardrails(self):
        before = self.report([1.0, 0.0], [10, 10])
        after = self.report([1.0, 1.0], [11, 11])
        decision = SelectionPolicy(SelectionConfig(min_delta=.01, max_regressions=0)).compare(before, after)
        self.assertTrue(decision.accepted)
        with self.assertRaises(ValueError):
            SelectionPolicy().compare(before, self.report([1.0]))
        self.assertFalse(SelectionPolicy().compare(before, self.report([0.0, 1.0])).accepted)
        self.assertFalse(SelectionPolicy().compare(before, self.report([1.0, 1.0], [100, 100])).accepted)


class EndToEndTests(unittest.TestCase):
    def test_optimizer_runs_with_project_runner_and_persists_all_artifacts(self):
        from examples.prompt_optimizer_example import make_adapter, make_generator
        with tempfile.TemporaryDirectory() as temporary:
            state = PromptState((PromptComponent("system.core", "system", "Answer quickly."),
                                 PromptComponent("policy.safety", "system", "Keep secrets private", editable=False)))
            train = [{"id": "train-1", "question": "2+2", "reference_answer": "4"},
                     {"id": "train-2", "question": "1+1", "reference_answer": "2"}]
            valid = [{"id": "val-1", "question": "3+3", "reference_answer": "6"},
                     {"id": "val-2", "question": "4+4", "reference_answer": "8"}]
            optimizer = PromptOptimizer(
                RunnerEvaluationHarness(make_adapter), make_generator(),
                store=ExperimentStore(temporary),
                config=OptimizationConfig(max_iterations=2, candidates_per_iteration=1),
            )
            result = optimizer.optimize(state, train, valid)
            self.assertEqual(result.baseline_validation_score, 0)
            self.assertEqual(result.selected_validation_score, 1)
            self.assertEqual(result.accepted_iterations, 1)
            self.assertEqual(result.stop_reason, "no_failures")
            self.assertTrue((result.session_dir / "iterations" / "001" / "diagnosis.json").exists())
            self.assertTrue((result.session_dir / "versions" / f"{result.selected_version_id}.json").exists())
            self.assertEqual(len(list((result.session_dir / "agent_runs").iterdir())), 4)
            session = json.loads((result.session_dir / "session.json").read_text())
            self.assertEqual(session["status"], "completed")
            attempts = json.loads((result.session_dir / "iterations" / "001" / "candidates.json").read_text())
            self.assertEqual(attempts[0]["status"], "accepted")

    def test_not_prompt_fixable_if_only_runner_errors(self):
        from app.runner import CallableAdapter
        class BrokenAgent:
            def ask(self, question):
                raise RuntimeError("tool unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            state = PromptState((PromptComponent("system.core", "system", "Do your best"),))
            class Generator:
                def generate(self, state, lesson):
                    raise AssertionError("LLM must not be called for execution errors")
            result = PromptOptimizer(RunnerEvaluationHarness(lambda state, seed: CallableAdapter(BrokenAgent)),
                                     Generator(), store=ExperimentStore(temporary)).optimize(
                                         state, [{"id": "train-1", "question": "hi", "reference_answer": "x"}],
                                         [{"id": "val-1", "question": "hi", "reference_answer": "x"}])
            self.assertEqual(result.stop_reason, "not_prompt_fixable")

    def test_rejected_candidates_are_recorded(self):
        from examples.prompt_optimizer_example import make_adapter
        class NoOpGenerator:
            def generate(self, state, lesson):
                component = state.get(lesson.component_id)
                return PromptPatch(component.component_id, "REPLACE_COMPONENT", digest(component.template),
                                   "Answer quickly.", "No-op", lesson.evidence_refs)
        with tempfile.TemporaryDirectory() as temporary:
            initial = PromptState((PromptComponent("system.core", "system", "Answer quickly."),))
            task = [{"id": "train-1", "question": "2+2", "reference_answer": "4"}]
            val = [{"id": "val-1", "question": "3+3", "reference_answer": "6"}]
            result = PromptOptimizer(RunnerEvaluationHarness(make_adapter), NoOpGenerator(),
                                     store=ExperimentStore(temporary),
                                     config=OptimizationConfig(max_iterations=1, candidates_per_iteration=1)).optimize(initial, task, val)
            self.assertEqual(result.stop_reason, "no_accepted_candidate")
            candidates = json.loads((result.session_dir / "iterations" / "001" / "candidates.json").read_text())
            self.assertEqual(candidates[0]["status"], "rejected")


if __name__ == "__main__":
    unittest.main()
