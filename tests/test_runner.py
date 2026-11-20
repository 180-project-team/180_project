"""Runner tests: no model downloads, MADQA install or extra packages."""

import json
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from app.runner import AgentRunner, CallableAdapter, SubprocessJSONLAdapter, load_questions
from app.runner.adapters.madqa import MADQAAdapter

ROOT = Path(__file__).resolve().parents[1]
ECHO = ROOT / "examples" / "echo_agent.py"


class FakeAgent:
    def __init__(self, events):
        self.events = events

    def start(self):
        self.events.append("start")

    def ask(self, q):
        self.events.append(("ask", q))
        if q == "bad":
            raise ValueError("question failure")
        return {"answer": q.upper(), "trajectory": ["step"]}

    def close(self):
        self.events.append("stop")


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runs = Path(self.temp.name) / "runs"
        self.events = []

    def factory(self):
        return CallableAdapter(lambda: FakeAgent(self.events))

    def runner(self, factory=None):
        return AgentRunner(factory or self.factory, runs_root=self.runs)

    def test_persists_separate_run_directory_with_all_artifacts(self):
        report = self.runner().run([
            {"id": "a", "question": "first", "answer_variants": [["FIRST"]], "evidence": ["doc"]},
            {"question": "bad"},
            {"id": "c", "question": "last"},
        ], metadata={"prompt_version": "v1"})
        self.assertEqual(report.status, "completed_with_errors")
        self.assertEqual((report.processed_count, report.failed_count), (3, 1))
        self.assertEqual(self.events, ["start", ("ask", "first"), ("ask", "bad"), ("ask", "last"), "stop"])
        self.assertTrue((report.run_dir / "artifacts").is_dir())
        self.assertTrue(report.events_path.exists())
        self.assertEqual([json.loads(x) for x in report.questions_path.read_text().splitlines()][0]["question"], "first")
        rows = report.load_results()
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1]["id"], "2")
        self.assertEqual(rows[0]["answer_variants"], [["FIRST"]])
        self.assertEqual(rows[0]["evidence"], ["doc"])
        self.assertEqual(rows[1]["status"], "error")
        meta = json.loads(report.metadata_path.read_text())
        self.assertEqual(meta["status"], "completed_with_errors")
        self.assertEqual(meta["metadata"], {"prompt_version": "v1"})
        self.assertTrue(meta["finished_at"])

    def test_no_evaluator_or_llm_in_runner_contract(self):
        report = self.runner().run([{"question": "yes"}])
        self.assertEqual(report.load_results()[0]["answer"], "YES")
        # In the project, an evaluator can receive the runner and invoke it.
        class ExampleEvaluator:
            def __init__(self, runner):
                self.runner = runner
            def evaluate(self):
                return self.runner.run([{"question": "second"}]).run_dir
        self.assertNotEqual(ExampleEvaluator(self.runner()).evaluate(), report.run_dir)

    def test_stop_if_start_fails_and_keep_failed_folder(self):
        class Broken(FakeAgent):
            def start(self):
                self.events.append("start")
                raise RuntimeError("bad startup")
        with self.assertRaisesRegex(RuntimeError, "bad startup"):
            self.runner(lambda: CallableAdapter(lambda: Broken(self.events))).run([{"question": "q"}])
        self.assertEqual(self.events, ["start", "stop"])
        folders = list(self.runs.iterdir())
        self.assertEqual(len(folders), 1)
        meta = json.loads((folders[0] / "metadata.json").read_text())
        self.assertEqual(meta["status"], "failed")
        self.assertIn("bad startup", meta["fatal_error"])
        self.assertTrue((folders[0] / "questions.jsonl").exists())
        self.assertTrue((folders[0] / "results.jsonl").exists())

    def test_validates_before_creating_dir_or_start(self):
        with self.assertRaises(ValueError):
            self.runner().run([{"question": ""}])
        self.assertFalse(self.runs.exists())
        self.assertEqual(self.events, [])

    def test_empty_batch_creates_completed_run_folder(self):
        report = self.runner().run([])
        self.assertEqual(report.load_results(), [])
        self.assertEqual(json.loads(report.metadata_path.read_text())["question_count"], 0)
        self.assertEqual(self.events, [])

    def test_fail_fast_writes_partial_results(self):
        report = self.runner().run([{"question": "bad"}, {"question": "good"}], continue_on_error=False)
        self.assertEqual(report.processed_count, 1)
        self.assertEqual(len(report.load_results()), 1)
        self.assertEqual(self.events[-1], "stop")

    def test_direct_string_response(self):
        class Simple:
            def ask(self, q):
                return "ok:" + q
        report = self.runner(lambda: CallableAdapter(Simple)).run([{"question": "ping"}])
        self.assertEqual(report.load_results()[0]["answer"], "ok:ping")

    def test_input_answer_does_not_mask_prediction(self):
        report = self.runner().run([{"question": "hello", "answer": "gold"}])
        self.assertEqual(report.load_results()[0]["answer"], "HELLO")

    def test_new_folder_per_run(self):
        runner = self.runner()
        first = runner.run([{"question": "a"}])
        second = runner.run([{"question": "a"}])
        self.assertNotEqual(first.run_dir, second.run_dir)
        self.assertTrue(first.run_dir.exists())
        self.assertTrue(second.run_dir.exists())

    def test_explicit_run_id_collision_and_path_traversal_rejected(self):
        runner = self.runner()
        runner.run([], run_id="my_run")
        with self.assertRaises(FileExistsError):
            runner.run([], run_id="my_run")
        for value in ("../bad", "/tmp/bad", "..", "has space"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.run([], run_id=value)

    def test_subprocess_agent_lifecycle_and_second_run(self):
        adapters = []
        def factory():
            adapter = SubprocessJSONLAdapter([sys.executable, "-u", str(ECHO)], startup_timeout=3)
            adapters.append(adapter)
            return adapter
        runner = self.runner(factory)
        for _ in range(2):
            report = runner.run([{"id": "X", "question": "hello"}])
            self.assertEqual(report.load_results()[0]["answer"], "HELLO")
        self.assertEqual(len(adapters), 2)
        self.assertTrue(all(a.process is None for a in adapters))
        self.assertTrue((report.run_dir / "artifacts" / "agent.stderr.log").exists())

    def test_subprocess_agent_persists_stderr_log(self):
        code = (
            "import json,sys; "
            "print('agent diagnostic',file=sys.stderr,flush=True); "
            "print(json.dumps({'ready':True}),flush=True); "
            "line=json.loads(input()); "
            "print(json.dumps({'id':line['id'],'answer':'ok'}),flush=True)"
        )
        report = self.runner(lambda: SubprocessJSONLAdapter([sys.executable, "-u", "-c", code], startup_timeout=3)).run([{"question": "x"}])
        log = (report.run_dir / "artifacts" / "agent.stderr.log").read_text()
        self.assertIn("agent diagnostic", log)

    def test_subprocess_worker_rejects_bad_startup(self):
        code = "import json; print(json.dumps({'ready':False,'error':'broken'}), flush=True)"
        factory = lambda: SubprocessJSONLAdapter([sys.executable, "-u", "-c", code], startup_timeout=3)
        with self.assertRaisesRegex(RuntimeError, "broken"):
            self.runner(factory).run([{"question": "hello"}])

    def test_subprocess_worker_times_out_and_stops(self):
        code = "import json,time; print(json.dumps({'ready':True}),flush=True); input(); time.sleep(4)"
        adapter = SubprocessJSONLAdapter([sys.executable, "-u", "-c", code], startup_timeout=3, question_timeout=0.2)
        report = self.runner(lambda: adapter).run([{"question": "hello"}])
        self.assertEqual(report.failed_count, 1)
        self.assertIn("TimeoutError", report.load_results()[0]["error"])
        self.assertIsNone(adapter.process)

    def test_subprocess_worker_rejects_protocol_mismatch(self):
        code = "import json; print(json.dumps({'ready':True}),flush=True); input(); print(json.dumps({'id':999,'answer':'ok'}),flush=True)"
        report = self.runner(lambda: SubprocessJSONLAdapter([sys.executable, "-u", "-c", code], startup_timeout=3)).run([{"question": "hi"}])
        self.assertEqual(report.failed_count, 1)
        self.assertIn("mismatch", report.load_results()[0]["error"])

    def test_subprocess_worker_no_stdout_logs_allowed(self):
        code = "print('debug',flush=True)"
        with self.assertRaisesRegex(RuntimeError, "non-JSON"):
            self.runner(lambda: SubprocessJSONLAdapter([sys.executable, "-u", "-c", code], startup_timeout=3)).run([{"question": "hi"}])

    def test_madqa_adapter_with_stubbed_upstream_module(self):
        root = Path(self.temp.name)
        (root / "utils.py").write_text(
            "class WhooshSearchEngine:\n"
            "    def __init__(self, source): self.source = source\n", encoding="utf-8")
        (root / "vllm_agent.py").write_text(
            "from utils import WhooshSearchEngine\n"
            "class VLLMSearchAgent:\n"
            "    def __init__(self, engine, model, base_url, api_key): self.engine = engine\n"
            "    def answer_question(self, question, max_iterations, top_k):\n"
            "        print('ordinary debug log should NOT break stdout JSON protocol')\n"
            "        return {'answer':[question.upper()], 'citations':[], 'trajectory':[], 'iterations':1}\n",
            encoding="utf-8")
        (root / "ocr.jsonl").write_text("{}\n", encoding="utf-8")
        adapter = MADQAAdapter(agent_dir=root, ocr_file="ocr.jsonl", model="stub", startup_timeout=3)
        report = self.runner(lambda: adapter).run([{"question": "hello", "answer_variants": [["HELLO"]]}])
        self.assertEqual(report.load_results()[0]["answer"], ["HELLO"])
        self.assertEqual(report.load_results()[0]["iterations"], 1)
        self.assertEqual(report.failed_count, 0)
        self.assertIsNone(adapter.process)

    def test_madqa_validates_settings_without_madqa_dependency(self):
        with self.assertRaises(ValueError):
            MADQAAdapter(agent_dir=self.temp.name, model="m", top_k=0)

    def test_madqa_missing_repo_fails_cleanly(self):
        adapter = MADQAAdapter(agent_dir=self.temp.name, model="m", startup_timeout=3)
        with self.assertRaisesRegex(RuntimeError, "MADQA files"):
            self.runner(lambda: adapter).run([{"question": "hi"}])
        self.assertIsNone(adapter.process)

    def test_app_module_cli_end_to_end(self):
        proc = subprocess.run([
            sys.executable, "-m", "app",
            "--adapter", "jsonl",
            "--agent-command", f'"{sys.executable}" -u "{ECHO}"',
            "--questions", str(ROOT / "examples" / "questions.jsonl"),
            "--runs-root", str(self.runs),
        ], cwd=ROOT, text=True, capture_output=True, timeout=10)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report_dir = Path(proc.stdout.strip())
        self.assertTrue(report_dir.is_dir())
        rows = [json.loads(line) for line in (report_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertTrue(rows)
        self.assertEqual(rows[0]["answer"], rows[0]["question"].upper())
        self.assertEqual(len(rows), 2)

    def test_question_loading(self):
        inp = Path(self.temp.name) / "questions.jsonl"
        inp.write_text('{"id":"x","question":"first"}\n"second"\n', encoding="utf-8")
        self.assertEqual(load_questions(inp, limit=1)[0]["id"], "x")
        self.assertEqual(load_questions(inp)[1]["question"], "second")
        inp.write_text('{"question":""}\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            load_questions(inp)


if __name__ == "__main__":
    unittest.main()
