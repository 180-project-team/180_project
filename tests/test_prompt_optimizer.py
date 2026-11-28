"""Prompt optimizer unit and end-to-end tests; no external model required."""

import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from app.prompt_optimizer import (
    LLMPatchGenerator, PromptComponent, PromptGuard, PromptPatch,
    PromptRenderer, PromptState, SelectionConfig, SelectionPolicy, digest,
    load_evaluation_report, propose_candidate,
)
from app.prompt_optimizer.evaluation_report import EvaluationItem, EvaluationReport
from app.prompt_optimizer.models import PromptLesson


class FixturePromptGenerator:
    def generate(self, state, lesson):
        component = state.get(lesson.component_id)
        return PromptPatch(
            component_id=lesson.component_id, operation="APPEND_RULE",
            before_hash=digest(component.template),
            payload="Verify the final answer against available information.",
            rationale="Follow the supplied advice.", evidence_refs=lesson.evidence_refs,
            rule_id="verify_answer",
        )

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
        lesson = PromptLesson("incorrect_answer-1", "system.core", "Check evidence", ("run#1",))
        patch = LLMPatchGenerator(FakeLLM()).generate(self.state, lesson)
        self.assertEqual(patch.operation, "APPEND_RULE")
        self.assertEqual(patch.component_id, "system.core")
        self.assertEqual(patch.before_hash, digest("Answer {question}"))


class ProposalTests(unittest.TestCase):
    def setUp(self):
        self.state = PromptState((
            PromptComponent("system.core", "system", "Answer the question."),
            PromptComponent("verifier.instructions", "system", "Check your answer."),
            PromptComponent("policy.safety", "system", "Keep data private", editable=False),
        ))
        self.diagnosis = {
            "run_id": "run-1", "case_id": "q1", "status": "FAILURE_FOUND",
            "failure_type": "UNSUPPORTED_ANSWER", "suggested_scope": "PROMPT",
            "hypotheses": [{"hypothesis_id": "verify_answer", "scope": "PROMPT", "advice": "Verify the final answer."}],
            "evidence_refs": [{"path": ["trajectory", 0]}],
        }

    def test_uses_existing_diagnosis_to_choose_an_editable_component(self):
        self.diagnosis["signals"] = [{
            "failure_type": "UNSUPPORTED_ANSWER",
            "component_id": "verifier.instructions",
        }]
        candidate, patch = propose_candidate(self.state, self.diagnosis, FixturePromptGenerator())
        self.assertEqual(patch.component_id, "verifier.instructions")
        self.assertEqual(patch.evidence_refs, ("run-1#q1:trajectory.0",))
        self.assertEqual(candidate.parent_id, self.state.prompt_version_id)
        self.assertEqual(candidate.get("system.core"), self.state.get("system.core"))
        self.assertEqual(candidate.get("policy.safety"), self.state.get("policy.safety"))

    def test_skips_structural_diagnosis(self):
        class UnusedGenerator:
            def generate(self, state, lesson):
                raise AssertionError("A structural diagnosis should not invoke the prompt generator")

        self.diagnosis["suggested_scope"] = "STRUCTURAL"
        self.diagnosis["hypotheses"] = [{
            "hypothesis_id": "retry_tool", "scope": "STRUCTURAL", "advice": "Retry the failed tool.",
        }]
        self.assertIsNone(propose_candidate(self.state, self.diagnosis, UnusedGenerator()))

    def test_uses_secondary_prompt_advice_unless_primary_scope_only_is_requested(self):
        self.diagnosis["suggested_scope"] = "STRUCTURAL"
        self.assertIsNotNone(propose_candidate(self.state, self.diagnosis, FixturePromptGenerator()))
        self.assertIsNone(propose_candidate(self.state, self.diagnosis, FixturePromptGenerator(), primary_scope_only=True))

    def test_skips_diagnostic_error_and_insufficient_data(self):
        for status in ("DIAGNOSTIC_ERROR", "INSUFFICIENT_DATA"):
            with self.subTest(status=status):
                self.diagnosis["status"] = status
                self.assertIsNone(propose_candidate(self.state, self.diagnosis, FixturePromptGenerator()))

    def test_rejects_no_op(self):
        class NoOpGenerator:
            def generate(self, state, lesson):
                component = state.get(lesson.component_id)
                return PromptPatch(
                    component.component_id, "REPLACE_COMPONENT", digest(component.template),
                    component.template, "No-op", lesson.evidence_refs,
                )

        with self.assertRaisesRegex(ValueError, "non-empty change"):
            propose_candidate(self.state, self.diagnosis, NoOpGenerator())


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
    def test_previous_demo_entry_point_uses_the_file_based_interface(self):
        from examples.prompt_optimizer_example import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--output", str(root)]), 0)
            manifest = json.loads((root / "patch.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "proposed")
            self.assertEqual(manifest["diagnosis_ref"]["case_id"], "demo-1")
            self.assertTrue((root / "candidate_code.txt").is_file())
            self.assertIsNotNone(manifest["patch"])

    def test_rejects_an_evaluator_report_for_another_question(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            answers = root / "runner.jsonl"
            scores = root / "evaluator.jsonl"
            answers.write_text(json.dumps({"id": "q1"}) + "\n", encoding="utf-8")
            scores.write_text(
                json.dumps({"id": "q2", "status": "ok", "fact_recall": 1}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "same non-empty set of task ids"):
                load_evaluation_report(answers, scores, prompt_version_id="v0", split="validation")



if __name__ == "__main__":
    unittest.main()
