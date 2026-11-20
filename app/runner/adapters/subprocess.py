"""Isolated JSON-lines subprocess adapter for agents in any language."""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence


class SubprocessJSONLAdapter:
    """An arbitrary executable agent using the newline-delimited JSON protocol.

    On startup, agent writes {"ready": true} on stdout.
    For each stdin {"id": N, "question": "..."}, it writes one stdout JSON
    object {"id": N, "answer": ...}, optionally with citations/trajectory.
    Logs MUST go to stderr; stdout is protocol-only. See examples/echo_agent.py.

    The adapter owns the process (and its POSIX process group). No `shell=True`.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        startup_timeout: float = 90.0,
        question_timeout: float = 300.0,
        shutdown_timeout: float = 5.0,
    ) -> None:
        if not command or isinstance(command, (str, bytes)):
            raise ValueError("command must be a non-empty argument sequence")
        if min(startup_timeout, question_timeout, shutdown_timeout) <= 0:
            raise ValueError("timeouts must be positive")
        self.command = list(command)
        self.cwd = Path(cwd) if cwd else None
        self.env = dict(env) if env is not None else None
        self.startup_timeout = startup_timeout
        self.question_timeout = question_timeout
        self.shutdown_timeout = shutdown_timeout
        self.process: subprocess.Popen[str] | None = None
        self._log = None
        self._log_path: Path | None = None
        self._messages: queue.Queue[str | None] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._next_id = 0
        self._broken = False

    def set_artifacts_dir(self, directory: str | Path) -> None:
        """Optional runner hook: persist agent stderr under the run directory."""
        if self.process is not None:
            raise RuntimeError("Cannot change artifacts directory while running")
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self._log_path = path / "agent.stderr.log"

    def start(self) -> None:
        if self.process is not None:
            raise RuntimeError("Adapter already started")
        self._log = (
            self._log_path.open("w+", encoding="utf-8")
            if self._log_path is not None
            else tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
        )
        extra: dict[str, Any] = {}
        if os.name == "posix":
            extra["start_new_session"] = True
        elif os.name == "nt":
            extra["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            self.process = subprocess.Popen(
                self.command,
                cwd=str(self.cwd) if self.cwd else None,
                env={**os.environ, **self.env} if self.env else None,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._log,
                text=True,
                encoding="utf-8",
                bufsize=1,
                **extra,
            )
            self._reader = threading.Thread(target=self._read_output, args=(self.process.stdout,), daemon=True)
            self._reader.start()
            ready = self._read_json(self.startup_timeout)
            if ready.get("ready") is not True:
                raise RuntimeError(f"Agent failed during startup: {ready.get('error', ready)}")
        except BaseException:
            self.stop()
            raise

    def _read_output(self, output: Any) -> None:
        try:
            for line in output:
                self._messages.put(line)
        finally:
            self._messages.put(None)

    def _log_tail(self) -> str:
        if self._log is None:
            return ""
        self._log.flush()
        self._log.seek(0, os.SEEK_END)
        end = self._log.tell()
        self._log.seek(max(0, end - 3000))
        return self._log.read()

    def _read_json(self, timeout: float) -> dict[str, Any]:
        try:
            line = self._messages.get(timeout=timeout)
        except queue.Empty as exc:
            self._broken = True
            raise TimeoutError(
                f"No agent response within {timeout}s; stderr tail:\n{self._log_tail()}"
            ) from exc
        if line is None:
            self._broken = True
            returncode = self.process.poll() if self.process else "?"
            raise RuntimeError(f"Agent process ended (code={returncode}); stderr tail:\n{self._log_tail()}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            self._broken = True
            raise RuntimeError(f"Agent printed non-JSON on stdout: {line[:200]!r}") from exc
        if not isinstance(value, dict):
            self._broken = True
            raise RuntimeError("Agent protocol response must be a JSON object")
        return value

    def ask(self, question: str) -> dict[str, Any]:
        if self._broken:
            raise RuntimeError("Agent protocol is broken; restart adapter")
        proc = self.process
        if proc is None or proc.poll() is not None or proc.stdin is None:
            raise RuntimeError("Agent process is not running")
        self._next_id += 1
        request_id = self._next_id
        try:
            proc.stdin.write(json.dumps({"id": request_id, "question": question}, ensure_ascii=False) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(f"Could not send question to agent: {exc}") from exc
        response = self._read_json(self.question_timeout)
        if response.get("id") != request_id:
            self._broken = True
            raise RuntimeError(f"Agent request id mismatch: expected {request_id}, got {response.get('id')!r}")
        return {key: value for key, value in response.items() if key != "id"}

    def stop(self) -> None:
        proc, self.process = self.process, None
        try:
            if proc is not None:
                # Do not wait indefinitely on an agent that ignores stdin EOF.
                if proc.stdin is not None:
                    try:
                        proc.stdin.close()
                    except OSError:
                        pass
                if proc.poll() is None:
                    try:
                        proc.wait(timeout=min(self.shutdown_timeout, 0.5))
                    except subprocess.TimeoutExpired:
                        pass
                # Signal the entire group we created, including workers spawned
                # by the agent. Never touch an externally managed process.
                if os.name == "posix":
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                elif proc.poll() is None:
                    proc.terminate()
                try:
                    proc.wait(timeout=self.shutdown_timeout)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    elif proc.poll() is None:
                        proc.kill()
                    proc.wait(timeout=self.shutdown_timeout)
                if proc.stdout is not None:
                    proc.stdout.close()
        finally:
            self._reader = None
            if self._log is not None:
                self._log.close()
                self._log = None
