"""Shared LLM client: HTTP protocol and lifecycle verified with a stub server."""

import json
import socket
import sys
import tempfile
import unittest
from pathlib import Path

from app.llm import LLMService, ManagedModelServer
from app.llm.server import is_ready


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


SERVER_STUB = '''
import json, sys
from http.server import HTTPServer, BaseHTTPRequestHandler
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == '/v1/models' else 404)
        self.end_headers()
    def do_POST(self):
        size = int(self.headers.get('Content-Length', '0'))
        body = json.loads(self.rfile.read(size))
        if self.path != '/v1/chat/completions':
            self.send_response(404); self.end_headers(); return
        answer = {'choices': [{'message': {'role': 'assistant', 'content': body['messages'][-1]['content'].upper()}}], 'model': body['model'], 'usage': {'total_tokens': 3}}
        data = json.dumps(answer).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def log_message(self, *args): pass
HTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()
'''


class LLMTests(unittest.TestCase):
    def setUp(self):
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}/v1"
        self.command = [sys.executable, "-u", "-c", SERVER_STUB, str(self.port)]

    def test_owned_server_starts_serves_and_stops(self):
        with LLMService(model="test-model", base_url=self.url, command=self.command, startup_timeout=4) as llm:
            self.assertTrue(is_ready(self.url + "/models"))
            self.assertEqual(llm.ask("hello"), "HELLO")
            self.assertEqual(llm.chat([{"role": "system", "content": "rule"}, {"role": "user", "content": "hi"}]), "HI")
            self.assertEqual(llm.chat_response([{"role": "user", "content": "sample"}])["usage"]["total_tokens"], 3)
        self.assertFalse(is_ready(self.url + "/models", timeout=0.2))

    def test_external_server_not_stopped_by_client(self):
        with ManagedModelServer(command=self.command, health_url=self.url + "/models", startup_timeout=4):
            with LLMService(model="test-model", base_url=self.url) as llm:
                self.assertEqual(llm.ask("go"), "GO")
            self.assertTrue(is_ready(self.url + "/models"))
        self.assertFalse(is_ready(self.url + "/models", timeout=0.2))

    def test_start_required_and_input_validation(self):
        client = LLMService(model="test-model", base_url=self.url)
        with self.assertRaisesRegex(RuntimeError, "Start LLMService"):
            client.chat([{"role": "user", "content": "hello"}])
        with self.assertRaises(ValueError):
            client.ask("")
        with self.assertRaises(ValueError):
            LLMService(model="", base_url=self.url)
        with self.assertRaises(ValueError):
            LLMService(model="m", base_url="file:///tmp")

    def test_unavailable_external_server_fails(self):
        client = LLMService(model="test-model", base_url=self.url)
        with self.assertRaises(RuntimeError):
            client.start()


if __name__ == "__main__":
    unittest.main()
