"""Batch execution and adapter tests."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from textwrap import dedent

from app.runner import AgentRunner, CallableAdapter, SubprocessJSONLAdapter, load_questions

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

    def run_batch(self, questions, factory=None, **options):
        runner = AgentRunner(lambda _: (factory or self.factory)())
        return list(runner.run(questions, self.runs, **options))

    def test_run_dir_and_results(self):
        seen_dirs = []
        def factory(run_dir):
            seen_dirs.append(run_dir)
            return self.factory()
        rows = list(AgentRunner(factory).run([
            {"question": "first", "answer": "gold", "evidence": ["doc"]},
            {"question": "bad"}, {"question": "last"},
        ], self.runs))
        self.assertEqual(seen_dirs, [self.runs])
        self.assertFalse(self.runs.exists())
        self.assertEqual([r["status"] for r in rows], ["ok", "error", "ok"])
        self.assertEqual(rows[0]["answer"], "FIRST")
        self.assertEqual(rows[0]["evidence"], ["doc"])
        self.assertEqual(rows[0]["trajectory"], ["step"])
        self.assertEqual(rows[1]["id"], "2")
        self.assertEqual(self.events, ["start", ("ask", "first"), ("ask", "bad"), ("ask", "last"), "stop"])

    def test_stop_if_start_fails(self):
        class Broken(FakeAgent):
            def start(self):
                raise RuntimeError("bad startup")
        self.runs.mkdir()
        results_path = self.runs / "results.jsonl"
        results_path.write_text('{"id":"previous"}\n', encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "bad startup"):
            self.run_batch(
                [{"question": "q"}],
                lambda: CallableAdapter(lambda: Broken(self.events)),
                results_path=results_path,
            )
        self.assertEqual(self.events, ["stop"])
        self.assertEqual(results_path.read_text(encoding="utf-8"), '{"id":"previous"}\n')
        self.assertEqual(list(self.runs.iterdir()), [results_path])

    def test_validation_and_empty_batch(self):
        with self.assertRaises(ValueError):
            self.run_batch([{"question": "q"}, {"question": ""}])
        self.assertEqual(self.run_batch([]), [])
        self.assertEqual(self.events, [])

    def test_fail_fast(self):
        rows = self.run_batch([{"question": "bad"}, {"question": "good"}], continue_on_error=False)
        self.assertEqual([r["status"] for r in rows], ["error"])
        self.assertEqual(self.events[-1], "stop")

    def test_results_path_contains_exactly_the_rows_from_the_last_completed_run(self):
        path = self.runs / "nested" / "agent.jsonl"
        questions = [{"question": "Кто?", "reference_answer": "Test"}, {"question": "bad"}]
        for _ in range(2):
            rows = self.run_batch(questions, results_path=path)
            written = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(written, rows)
        self.assertEqual([row["status"] for row in written], ["ok", "error"])
        self.assertIn("question failure", written[1]["error"])

    def test_direct_string_response(self):
        class Simple:
            def ask(self, q):
                return "ok:" + q
        rows = self.run_batch([{"question": "ping"}], lambda: CallableAdapter(Simple))
        self.assertEqual(rows[0]["answer"], "ok:ping")

    def test_subprocess_agent_lifecycle_and_second_run(self):
        adapters = []
        def factory():
            adapter = SubprocessJSONLAdapter([sys.executable, "-u", str(ECHO)], startup_timeout=3)
            adapters.append(adapter)
            return adapter
        runner = AgentRunner(lambda _: factory())
        for _ in range(2):
            rows = list(runner.run([{"id": "X", "question": "hello"}], self.runs))
            self.assertEqual(rows[0]["answer"], "HELLO")
        self.assertEqual(len(adapters), 2)
        self.assertTrue(all(a.process is None for a in adapters))
    def test_subprocess_worker_rejects_bad_startup(self):
        code = "import json; print(json.dumps({'ready':False,'error':'broken'}), flush=True)"
        factory = lambda: SubprocessJSONLAdapter([sys.executable, "-u", "-c", code], startup_timeout=3)
        with self.assertRaisesRegex(RuntimeError, "broken"):
            self.run_batch([{"question": "hello"}], factory)

    def test_subprocess_worker_times_out_and_stops(self):
        code = "import json,time; print(json.dumps({'ready':True}),flush=True); input(); time.sleep(4)"
        adapter = SubprocessJSONLAdapter([sys.executable, "-u", "-c", code], startup_timeout=3, question_timeout=0.2)
        rows = self.run_batch([{"question": "hello"}], lambda: adapter)
        self.assertEqual(rows[0]["status"], "error")
        self.assertIn("TimeoutError", rows[0]["error"])
        self.assertIsNone(adapter.process)

    def test_subprocess_worker_rejects_protocol_mismatch(self):
        code = "import json; print(json.dumps({'ready':True}),flush=True); input(); print(json.dumps({'id':999,'answer':'ok'}),flush=True)"
        rows = self.run_batch([{"question": "hi"}], lambda: SubprocessJSONLAdapter([sys.executable, "-u", "-c", code], startup_timeout=3))
        self.assertEqual(rows[0]["status"], "error")
        self.assertIn("mismatch", rows[0]["error"])

    def test_subprocess_worker_no_stdout_logs_allowed(self):
        code = "print('debug',flush=True)"
        with self.assertRaisesRegex(RuntimeError, "non-JSON"):
            self.run_batch([{"question": "hi"}], lambda: SubprocessJSONLAdapter([sys.executable, "-u", "-c", code], startup_timeout=3))


    def test_app_module_cli_end_to_end(self):
        proc = subprocess.run([
            sys.executable, "-m", "app",
            "--agent-command", f'"{sys.executable}" -u "{ECHO.name}"',
            "--agent-cwd", str(ECHO.parent),
            "--questions", str(ROOT / "examples" / "questions.jsonl"),
            "--run-dir", str(self.runs),
        ], cwd=ROOT, text=True, capture_output=True, timeout=10)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertFalse(self.runs.exists())
        rows = [json.loads(line) for line in proc.stdout.splitlines()]
        self.assertTrue(rows)
        self.assertEqual(rows[0]["answer"], rows[0]["question"].upper())
        self.assertEqual(len(rows), 2)

    def test_skim_cli_preserves_native_trajectory_and_reuses_index(self):
        temp_dir = Path(self.temp.name)
        corpus_path = temp_dir / "corpus.jsonl"
        corpus_path.write_text('{"docid":"doc1","text":"Документ"}\n', encoding="utf-8")
        (temp_dir / "agent_search.py").write_text(dedent('''\
            import os

            def as_units(docs):
                assert docs[0]["doc_id"] == "doc1"
                return docs

            class Agent:
                def index(self, units):
                    assert not hasattr(self, "calls")
                    self.calls = 0

                def search(self, question, k):
                    self.calls += 1
                    self.last_trajectory_meta = {
                        "final_answer": f"{self.calls}: {question}",
                        "trajectory": [{"action": "search_s", "args": {"query": question},
                                        "observation": "Document content"}],
                        "stopped": "answer",
                    }
                    return ["doc1"]

            def build_agent(strategy, **options):
                assert strategy == "sieve_bm25"
                assert options["model"] == "local-test"
                os.write(1, b"Skim diagnostic message\\n")
                return Agent()
            '''), encoding="utf-8")
        results_path = temp_dir / "results" / "runner_results.jsonl"
        proc = subprocess.run([
            sys.executable, "-m", "app", "--adapter", "skim",
            "--skim-python", sys.executable, "--skim-model", "local-test",
            "--skim-corpus", str(corpus_path),
            "--questions", str(ROOT / "examples" / "questions.jsonl"),
            "--run-dir", str(self.runs), "--results-path", str(results_path),
        ], cwd=ROOT, env={**os.environ, "PYTHONPATH": str(temp_dir)},
            text=True, capture_output=True, timeout=10)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows, [json.loads(line) for line in proc.stdout.splitlines()])
        self.assertEqual([row["answer"].split(":", 1)[0] for row in rows], ["1", "2"])
        self.assertEqual(rows[0]["trajectory"], [
            {"action": "search_s", "args": {"query": rows[0]["question"]}, "observation": "Document content"},
        ])
        self.assertEqual(rows[0]["ranking"], ["doc1"])
        self.assertEqual(rows[0]["stopped"], "answer")

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
