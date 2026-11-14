"""No MADQA, models, GPUs, or extra third-party libraries required."""

import json
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from app.runner import AgentRunner, CallableAdapter, SubprocessJSONLAdapter, load_questions
from app.runner.adapters.madqa import MADQAAdapter
from app.runner.infrastructure.model_server import ManagedModelServer, is_ready

ROOT = Path(__file__).resolve().parents[1]
ECHO = ROOT / "examples" / "echo_agent.py"


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


MODEL_STUB = '''
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == '/v1/models' else 404)
        self.end_headers()
    def log_message(self, *args):
        pass
HTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()
'''


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
        self.events = []

    def factory(self):
        return CallableAdapter(lambda: FakeAgent(self.events))

    def test_runs_one_agent_for_batch_preserving_metadata_and_evaluator(self):
        path = Path(self.temp.name) / "output.jsonl"
        def evaluator(rows):
            self.events.append("evaluator")
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[1]["status"], "error")
            return {"external_score": 1}
        report = AgentRunner(self.factory).run([
            {"id": "a", "question": "first", "answer_variants": [["FIRST"]], "evidence": ["doc"]},
            {"question": "bad"},
            {"id": "c", "question": "last"},
        ], evaluator=evaluator, output_path=path)
        self.assertEqual(report.failed_count, 1)
        self.assertEqual(report.evaluation, {"external_score": 1})
        self.assertEqual(report.results[1]["id"], "2")
        self.assertEqual(report.results[0]["answer_variants"], [["FIRST"]])
        self.assertEqual(report.results[0]["evidence"], ["doc"])
        self.assertEqual(self.events, [
            "start", ("ask", "first"), ("ask", "bad"), ("ask", "last"), "evaluator", "stop"
        ])
        self.assertEqual([json.loads(x) for x in path.read_text().splitlines()], report.results)

    def test_stop_on_evaluator_failure(self):
        def fail(_):
            raise RuntimeError("evaluator failed")
        with self.assertRaisesRegex(RuntimeError, "evaluator failed"):
            AgentRunner(self.factory).run([{"question": "ok"}], evaluator=fail)
        self.assertEqual(self.events[-1], "stop")

    def test_stop_if_start_fails(self):
        class Broken(FakeAgent):
            def start(self):
                self.events.append("start")
                raise RuntimeError("bad startup")
        with self.assertRaisesRegex(RuntimeError, "bad startup"):
            AgentRunner(lambda: CallableAdapter(lambda: Broken(self.events))).run([{"question": "q"}])
        self.assertEqual(self.events, ["start", "stop"])

    def test_validates_before_start(self):
        with self.assertRaises(ValueError):
            AgentRunner(self.factory).run([{"question": ""}])
        self.assertEqual(self.events, [])

    def test_empty_batch_skips_start(self):
        self.assertEqual(AgentRunner(self.factory).run([]).results, [])
        self.assertEqual(self.events, [])

    def test_fail_fast(self):
        report = AgentRunner(self.factory).run([{"question": "bad"}, {"question": "good"}], continue_on_error=False)
        self.assertEqual(len(report.results), 1)
        self.assertEqual(self.events[-1], "stop")

    def test_direct_string_response(self):
        class Simple:
            def ask(self, q):
                return "ok:" + q
        report = AgentRunner(lambda: CallableAdapter(Simple)).run([{"question": "ping"}])
        self.assertEqual(report.results[0]["answer"], "ok:ping")

    def test_input_answer_does_not_mask_prediction(self):
        report = AgentRunner(self.factory).run([{"question": "hello", "answer": "gold"}])
        self.assertEqual(report.results[0]["answer"], "HELLO")

    def test_subprocess_agent_lifecycle_and_second_run(self):
        holder = []
        def factory():
            adapter = SubprocessJSONLAdapter([sys.executable, "-u", str(ECHO)], startup_timeout=3)
            holder.append(adapter)
            return adapter
        runner = AgentRunner(factory)
        for _ in range(2):
            report = runner.run([{"id": "X", "question": "hello"}])
            self.assertEqual(report.results[0]["answer"], "HELLO")
            self.assertEqual(report.results[0]["citations"], [])
        self.assertEqual(len(holder), 2)
        self.assertTrue(all(a.process is None for a in holder))

    def test_subprocess_worker_rejects_bad_startup(self):
        code = "import json; print(json.dumps({'ready':False,'error':'broken'}), flush=True)"
        factory = lambda: SubprocessJSONLAdapter([sys.executable, "-u", "-c", code], startup_timeout=3)
        with self.assertRaisesRegex(RuntimeError, "broken"):
            AgentRunner(factory).run([{"question": "hello"}])

    def test_subprocess_worker_times_out_and_stops(self):
        code = "import json,time; print(json.dumps({'ready':True}),flush=True); input(); time.sleep(4)"
        adapter = SubprocessJSONLAdapter([sys.executable, "-u", "-c", code], startup_timeout=3, question_timeout=0.2)
        report = AgentRunner(lambda: adapter).run([{"question": "hello"}])
        self.assertEqual(report.failed_count, 1)
        self.assertIn("TimeoutError", report.results[0]["error"])
        self.assertIsNone(adapter.process)

    def test_subprocess_worker_rejects_protocol_mismatch(self):
        code = "import json; print(json.dumps({'ready':True}),flush=True); input(); print(json.dumps({'id':999,'answer':'ok'}),flush=True)"
        report = AgentRunner(lambda: SubprocessJSONLAdapter([sys.executable, "-u", "-c", code], startup_timeout=3)).run([{"question": "hi"}])
        self.assertEqual(report.failed_count, 1)
        self.assertIn("mismatch", report.results[0]["error"])

    def test_subprocess_worker_no_stdout_logs_allowed(self):
        code = "print('debug',flush=True)"
        with self.assertRaisesRegex(RuntimeError, "non-JSON"):
            AgentRunner(lambda: SubprocessJSONLAdapter([sys.executable, "-u", "-c", code], startup_timeout=3)).run([{"question": "hi"}])

    def test_model_service_launches_first_and_stops_last(self):
        port = free_port()
        url = f"http://127.0.0.1:{port}/v1/models"
        command = [sys.executable, "-u", "-c", MODEL_STUB, str(port)]
        service = ManagedModelServer(command=command, health_url=url, startup_timeout=4, shutdown_timeout=2)
        class CheckAgent(FakeAgent):
            def start(self):
                self.events.append("start")
                assert is_ready(url)
        def evaluator(_):
            assert is_ready(url)
            return {"external": True}
        report = AgentRunner(lambda: CallableAdapter(lambda: CheckAgent(self.events)), model_service=service).run(
            [{"question": "hello"}], evaluator=evaluator
        )
        self.assertEqual(report.evaluation, {"external": True})
        self.assertFalse(is_ready(url, timeout=0.2))

    def test_external_model_service_is_not_killed(self):
        port = free_port()
        url = f"http://127.0.0.1:{port}/v1/models"
        command = [sys.executable, "-u", "-c", MODEL_STUB, str(port)]
        with ManagedModelServer(command=command, health_url=url, startup_timeout=4, shutdown_timeout=2):
            check_only = ManagedModelServer(command=None, health_url=url, startup_timeout=4)
            AgentRunner(self.factory, model_service=check_only).run([{"question": "ok"}])
            self.assertTrue(is_ready(url))
            with self.assertRaisesRegex(RuntimeError, "already available"):
                AgentRunner(self.factory, model_service=ManagedModelServer(command=command, health_url=url)).run([{"question": "hi"}])
            self.assertTrue(is_ready(url))
        self.assertFalse(is_ready(url, timeout=0.2))

    def test_madqa_adapter_with_stubbed_upstream_module(self):
        """Check the real worker protocol, without downloading MADQA or a model."""
        root = Path(self.temp.name)
        (root / "utils.py").write_text(
            "class WhooshSearchEngine:\n"
            "    def __init__(self, source): self.source = source\n",
            encoding="utf-8",
        )
        (root / "vllm_agent.py").write_text(
            "from utils import WhooshSearchEngine\n"
            "class VLLMSearchAgent:\n"
            "    def __init__(self, engine, model, base_url, api_key): self.engine = engine\n"
            "    def answer_question(self, question, max_iterations, top_k):\n"
            "        print('ordinary debug log should NOT break stdout JSON protocol')\n"
            "        return {'answer':[question.upper()], 'citations':[], 'trajectory':[], 'iterations':1}\n",
            encoding="utf-8",
        )
        (root / "ocr.jsonl").write_text("{}\n", encoding="utf-8")
        adapter = MADQAAdapter(agent_dir=root, ocr_file="ocr.jsonl", model="stub", startup_timeout=3)
        report = AgentRunner(lambda: adapter).run([{"question": "hello", "answer_variants": [["HELLO"]]}])
        self.assertEqual(report.results[0]["answer"], ["HELLO"])
        self.assertEqual(report.results[0]["answer_variants"], [["HELLO"]])
        self.assertEqual(report.results[0]["iterations"], 1)
        self.assertEqual(report.failed_count, 0)
        self.assertIsNone(adapter.process)

    def test_madqa_validates_settings_without_madqa_dependency(self):
        with self.assertRaises(ValueError):
            MADQAAdapter(agent_dir=self.temp.name, model="m", top_k=0)

    def test_madqa_missing_repo_fails_cleanly(self):
        adapter = MADQAAdapter(agent_dir=self.temp.name, model="m", startup_timeout=3)
        with self.assertRaisesRegex(RuntimeError, "MADQA files"):
            AgentRunner(lambda: adapter).run([{"question": "hi"}])
        self.assertIsNone(adapter.process)

    def test_app_module_cli_end_to_end(self):
        result_file = Path(self.temp.name) / "results.jsonl"
        proc = subprocess.run([
            sys.executable, "-m", "app",
            "--adapter", "jsonl",
            "--agent-command", f'"{sys.executable}" -u "{ECHO}"',
            "--questions", str(ROOT / "examples" / "questions.jsonl"),
            "--output", str(result_file),
        ], cwd=ROOT, text=True, capture_output=True, timeout=10)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        rows = [json.loads(line) for line in result_file.read_text(encoding="utf-8").splitlines()]
        self.assertTrue(rows)
        self.assertEqual(rows[0]["answer"], rows[0]["question"].upper())
        self.assertEqual(json.loads(proc.stdout)["processed"], len(rows))

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
