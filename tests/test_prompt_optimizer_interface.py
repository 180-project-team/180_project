"""Verify L1 integration with real Diagnostics and a supplied source snapshot."""

import copy
import io
import json
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app.diagnostics import diagnose
from app.prompt_optimizer import PromptOptimizer, SourcePromptPatch
from app.prompt_optimizer.__main__ import main as cli_main
from app.prompt_optimizer.models import digest


BEFORE = 'SYSTEM_PROMPT = "Answer {question}."'
AFTER = 'SYSTEM_PROMPT = "Use the available evidence. Answer {question}."'
PYTHON_SOURCE = BEFORE + '\n\ndef answer(question):\n    return SYSTEM_PROMPT.format(question=question)\n'
AGENT_CODE = 'project/\n├── agent.py\n└── tools.py\n\n# agent.py\n' + PYTHON_SOURCE + '\n# tools.py\nMAX_STEPS = 10\n'


class FixedGenerator:
    def __init__(self, before=BEFORE, after=AFTER):
        self.before, self.after = before, after
        self.calls = []

    def generate(self, agent_code, reports):
        self.calls.append((agent_code, copy.deepcopy(reports)))
        return {
            "before": self.before, "after": self.after, "rationale": "Follow diagnostic advice.",
            "report_index": reports[0]["report_index"],
            "hypothesis_id": reports[0]["hypotheses"][0]["hypothesis_id"],
        }


class SourcePatchTests(unittest.TestCase):
    def make_patch(self, code=AGENT_CODE, before=BEFORE, after=AFTER):
        return SourcePromptPatch.create(
            code, before=before, after=after, rationale="Check the evidence.",
            report_index=0, hypothesis_id="check_evidence", evidence_refs=[{"path": ["answer"]}],
        )

    def test_patch_binds_to_bundle_and_preserves_agent_execution(self):
        result = self.make_patch().apply(AGENT_CODE)
        self.assertEqual(result, AGENT_CODE.replace(BEFORE, AFTER, 1))
        namespace = {}
        exec(self.make_patch(code=PYTHON_SOURCE).apply(PYTHON_SOURCE), namespace)
        self.assertEqual(namespace["answer"]("Who?"), "Use the available evidence. Answer Who?.")
        with self.assertRaisesRegex(ValueError, "Stale patch"):
            self.make_patch().apply(AGENT_CODE + "# changed")

    def test_rejects_logic_changes_and_non_prompt_literals(self):
        for before, after in (
            (BEFORE + "\nMAX_STEPS = 10", AFTER + "\nMAX_STEPS = 100"),
            ('API_URL = "https://example.com"', 'API_URL = "https://other.example.com"'),
        ):
            with self.subTest(before=before), self.assertRaises(ValueError):
                self.make_patch(code=before, before=before, after=after)

    def test_rejects_duplicate_fragments_and_missing_placeholders(self):
        with self.assertRaisesRegex(ValueError, "exactly once"):
            self.make_patch(code=BEFORE + "\n" + BEFORE)
        with self.assertRaisesRegex(ValueError, "template variables"):
            self.make_patch(after='SYSTEM_PROMPT = "Answer carefully."')

    def test_f_string_expressions_cannot_change(self):
        before = 'SYSTEM_PROMPT = f"Answer {question}."'
        after = 'SYSTEM_PROMPT = f"Check evidence. Answer {question}."'
        self.assertEqual(self.make_patch(code=before, before=before, after=after).apply(before), after)
        with self.assertRaisesRegex(ValueError, "template variables"):
            self.make_patch(code=before, before=before, after=after.replace("{question}", "{question.lower()}"))

    def test_indented_unicode_prompts_keep_indentation_and_source(self):
        before = '    SYSTEM_PROMPT = "Ответь на {question}."'
        after = '    SYSTEM_PROMPT = "Проверь факты. Ответь на {question}."'
        self.assertEqual(self.make_patch(code=before, before=before, after=after).apply(before), after)
        with self.assertRaisesRegex(ValueError, "only prompt text"):
            self.make_patch(code=before, before=before, after=after.lstrip())

    def test_chat_content_and_tool_descriptions_are_supported(self):
        for before, after in (
            ('messages = [{"role": "system", "content": "Answer carefully."}]',
             'messages = [{"role": "system", "content": "Use evidence. Answer carefully."}]'),
            ('register_tool(description="Search documents.")',
             'register_tool(description="Search relevant documents using concise queries.")'),
        ):
            with self.subTest(before=before):
                self.assertEqual(self.make_patch(code=before, before=before, after=after).apply(before), after)

    def test_comments_cannot_escape_the_prompt_boundary(self):
        with self.assertRaisesRegex(ValueError, "only prompt text"):
            self.make_patch(after=AFTER + " # hide remaining code")

    def test_examples_inside_documentation_and_comments_are_not_agent_prompts(self):
        for code in ('"""Example:\n' + BEFORE + '\n"""', 'f"""Example:\n' + BEFORE + '\n"""', '# ' + BEFORE):
            with self.subTest(code=code), self.assertRaises(ValueError):
                self.make_patch(code=code)

    def test_fragment_cannot_replace_part_of_a_larger_statement(self):
        before = 'instructions="Answer carefully."'
        with self.assertRaisesRegex(ValueError, "complete source lines"):
            self.make_patch(code=f"configure({before})", before=before, after='instructions="Use evidence."')


class FileInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.source = self.root / "diagnoses.jsonl"
        self.output = self.root / "optimizer"
        # The primary technical failure coexists with real secondary PROMPT advice.
        row = {
            "id": "mixed", "status": "ok", "question": "Who directed both films?", "answer": "Alice",
            "trajectory": [
                {"action": "bm25_search", "observation": "ERROR: Timeout"},
                {"action": "visit", "observation": "Alice directed the first film. Bob directed the second."},
            ],
            "metrics": {"fact_recall": 0.4, "retrieval_fact_recall": 1.0},
        }
        self.report = asdict(diagnose([row], run_id="run-1", version_id="v1").reports[0])
        self.write_reports([self.report])

    def write_reports(self, reports):
        self.source.write_text("".join(json.dumps(report) + "\n" for report in reports), encoding="utf-8")

    def manifest(self):
        return json.loads((self.output / "patch.json").read_text(encoding="utf-8"))

    def test_three_inputs_produce_a_patch_from_real_secondary_hypotheses(self):
        before = self.source.read_bytes()
        generator = FixedGenerator()
        result = PromptOptimizer(generator=generator).optimize(str(self.source), str(self.output), AGENT_CODE)

        self.assertEqual(result.status, "proposed")
        self.assertEqual(result.candidate_code, AGENT_CODE.replace(BEFORE, AFTER, 1))
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual({path.name for path in self.output.iterdir()}, {"patch.json", "candidate_code.txt", "changes.diff"})
        manifest = self.manifest()
        self.assertEqual(manifest["source_hash"], digest(AGENT_CODE))
        self.assertEqual(manifest["candidate_hash"], digest(result.candidate_code))
        self.assertEqual(manifest["diagnosis_ref"], {"case_id": "mixed", "run_id": "run-1", "version_id": "v1", "report_index": 0})
        self.assertEqual(manifest["patch"]["evidence_refs"], self.report["evidence_refs"])
        self.assertEqual((self.output / "candidate_code.txt").read_text(encoding="utf-8"), result.candidate_code)
        self.assertIn("Use the available evidence", (self.output / "changes.diff").read_text(encoding="utf-8"))
        sent = generator.calls[0][1]
        self.assertEqual(sent[0]["suggested_scope"], "STRUCTURAL")
        self.assertTrue(all(item["scope"] == "PROMPT" for item in sent[0]["hypotheses"]))

    def test_primary_only_mode_and_diagnostic_errors_skip_without_model_calls(self):
        generator = FixedGenerator()
        result = PromptOptimizer(generator=generator, primary_scope_only=True).optimize(self.source, self.output, AGENT_CODE)
        self.assertEqual(result.status, "skipped")
        self.assertEqual(generator.calls, [])
        for status in ("DIAGNOSTIC_ERROR", "INSUFFICIENT_DATA"):
            with self.subTest(status=status):
                self.write_reports([{**self.report, "status": status}])
                result = PromptOptimizer(generator=generator).optimize(self.source, self.output, AGENT_CODE)
                self.assertEqual(result.status, "skipped")
                self.assertEqual(generator.calls, [])

    def test_stable_report_indices_are_kept_when_earlier_cases_are_skipped(self):
        self.write_reports([{**self.report, "case_id": "skip", "status": "DIAGNOSTIC_ERROR"}, self.report])
        result = PromptOptimizer(generator=FixedGenerator()).optimize(self.source, self.output, AGENT_CODE)
        self.assertEqual(result.patch.report_index, 1)

    def test_rejected_patch_clears_a_previous_candidate(self):
        PromptOptimizer(generator=FixedGenerator()).optimize(self.source, self.output, AGENT_CODE)
        result = PromptOptimizer(generator=FixedGenerator(after=AFTER + "\nMAX_STEPS = 100")).optimize(self.source, self.output, AGENT_CODE)
        self.assertEqual(result.status, "rejected")
        self.assertIsNone(self.manifest()["patch"])
        self.assertEqual((self.output / "candidate_code.txt").read_text(encoding="utf-8"), AGENT_CODE)
        self.assertEqual((self.output / "changes.diff").read_text(encoding="utf-8"), "")

    def test_shared_llm_receives_source_and_advice_and_is_not_started_by_optimizer(self):
        class FakeLLM:
            def __init__(self):
                self.calls = []

            def chat(self, messages, **options):
                self.calls.append((messages, options))
                payload = json.loads(messages[1]["content"])
                return json.dumps(FixedGenerator().generate(payload["agent_code"], payload["reports"]))

        llm = FakeLLM()
        result = PromptOptimizer(llm).optimize(self.source, self.output, AGENT_CODE)
        self.assertEqual(result.status, "proposed")
        self.assertEqual(json.loads(llm.calls[0][0][1]["content"])["agent_code"], AGENT_CODE)
        self.assertEqual(llm.calls[0][1], {"temperature": 0})

    def test_foreign_hypothesis_and_invalid_model_json_are_rejected(self):
        class FakeLLM:
            def __init__(self, reply):
                self.reply = reply

            def chat(self, messages, **options):
                return self.reply

        for reply in ('not JSON', '{"report_index":0,"hypothesis_id":"invented"}'):
            with self.subTest(reply=reply):
                result = PromptOptimizer(FakeLLM(reply)).optimize(self.source, self.output, AGENT_CODE)
                self.assertEqual(result.status, "rejected")
                self.assertIsNone(result.patch)

    def test_generator_can_abstain_or_report_an_unavailable_model(self):
        class Abstain:
            def generate(self, agent_code, reports):
                return {"no_change": True, "reason": "Prompt is loaded dynamically."}

        class Unavailable:
            def generate(self, agent_code, reports):
                raise RuntimeError("Model unavailable")

        result = PromptOptimizer(generator=Abstain()).optimize(self.source, self.output, AGENT_CODE)
        self.assertEqual(result.status, "skipped")
        result = PromptOptimizer(generator=Unavailable()).optimize(self.source, self.output, AGENT_CODE)
        self.assertEqual(result.status, "error")
        self.assertIn("Model unavailable", result.reason)

    def test_generator_cannot_rewrite_the_supplied_hypothesis_allowlist(self):
        class MutatingGenerator(FixedGenerator):
            def generate(self, agent_code, reports):
                reports[0]["hypotheses"] = [{"hypothesis_id": "invented", "scope": "PROMPT", "advice": "Invented advice."}]
                return super().generate(agent_code, reports)

        result = PromptOptimizer(generator=MutatingGenerator()).optimize(self.source, self.output, AGENT_CODE)
        self.assertEqual(result.status, "rejected")
        self.assertIn("supplied PROMPT hypothesis", result.reason)

    def test_cli_accepts_the_source_string_and_a_source_text_file(self):
        code_file = self.root / "agent.txt"
        code_file.write_text(AGENT_CODE, encoding="utf-8")
        for code_args in (["--agent-code", AGENT_CODE], ["--agent-code-file", str(code_file)]):
            with self.subTest(code_args=code_args), patch(
                "app.prompt_optimizer.__main__.import_object", return_value=lambda: FixedGenerator(),
            ), redirect_stdout(io.StringIO()):
                self.assertEqual(cli_main([
                    "--diagnosis", str(self.source), "--output-dir", str(self.output),
                    "--generator", "test:factory", *code_args,
                ]), 0)
                self.assertEqual(self.manifest()["status"], "proposed")

    def test_input_cannot_be_overwritten_by_an_output_file(self):
        source = self.output / "patch.json"
        self.output.mkdir()
        source.write_text(json.dumps(self.report), encoding="utf-8")
        original = source.read_bytes()
        with self.assertRaisesRegex(ValueError, "overlaps"):
            PromptOptimizer(generator=FixedGenerator()).optimize(source, self.output, AGENT_CODE)
        self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
