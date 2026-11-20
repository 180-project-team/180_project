"""Own the lifecycle of an OpenAI-compatible local model server.

The shared LLM service NEVER terminates an already-running/unowned server.
"""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def is_ready(url: str, api_key: str | None = None, *, timeout: float = 1.5) -> bool:
    """Check the OpenAI-compatible /v1/models endpoint."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        with urlopen(Request(url, headers=headers), timeout=timeout) as response:
            return response.status == 200
    except (URLError, HTTPError, OSError, TimeoutError):
        return False


class ManagedModelServer:
    def __init__(
        self,
        *,
        command: Sequence[str] | None,
        health_url: str,
        api_key: str | None = None,
        cwd: Path | None = None,
        startup_timeout: float = 120.0,
        shutdown_timeout: float = 10.0,
        poll_interval: float = 0.25,
    ) -> None:
        if startup_timeout <= 0 or shutdown_timeout <= 0 or poll_interval <= 0:
            raise ValueError("Timeouts and poll_interval must be positive")
        self.command = list(command) if command is not None else None
        if self.command is not None and not self.command:
            raise ValueError("Model server command cannot be empty")
        self.health_url = health_url
        self.api_key = api_key
        self.cwd = Path(cwd) if cwd is not None else None
        self.startup_timeout = startup_timeout
        self.shutdown_timeout = shutdown_timeout
        self.poll_interval = poll_interval
        self.process: subprocess.Popen[bytes] | None = None
        self._log = None

    def __enter__(self) -> "ManagedModelServer":
        try:
            self.start()
            return self
        except BaseException:
            self.stop()
            raise

    def __exit__(self, *_: object) -> None:
        self.stop()

    def start(self) -> None:
        if self.command is None:
            if not is_ready(self.health_url, self.api_key):
                raise RuntimeError(
                    f"Existing model server is not available at {self.health_url}. "
                    "Supply a local command to start one, or start it manually."
                )
            return
        if is_ready(self.health_url, self.api_key):
            raise RuntimeError(
                f"A model server is already available at {self.health_url}. "
                "Refusing to start another process on the same endpoint. "
                "Connect without a command to use the existing server without owning it."
            )
        self._log = tempfile.TemporaryFile(mode="w+b")
        options: dict = {}
        if os.name == "posix":
            options["start_new_session"] = True
        elif os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        self.process = subprocess.Popen(
            self.command,
            cwd=str(self.cwd) if self.cwd is not None else None,
            stdin=subprocess.DEVNULL,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            **options,
        )
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"Model server exited with code {self.process.returncode}.\n"
                    f"Log tail:\n{self._log_tail()}"
                )
            if is_ready(self.health_url, self.api_key, timeout=0.5):
                return
            time.sleep(self.poll_interval)
        raise TimeoutError(
            f"Model server did not become ready at {self.health_url} "
            f"within {self.startup_timeout}s.\nLog tail:\n{self._log_tail()}"
        )

    def _log_tail(self) -> str:
        if self._log is None:
            return ""
        self._log.flush()
        self._log.seek(0, os.SEEK_END)
        size = self._log.tell()
        self._log.seek(max(0, size - 3000))
        return self._log.read().decode("utf-8", errors="replace")

    def stop(self) -> None:
        process = self.process
        self.process = None
        try:
            if process is not None:
                # On POSIX, terminate the *whole process group* (including child workers).
                # Only target our own process group created with start_new_session.
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                elif process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=self.shutdown_timeout)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    elif process.poll() is None:
                        process.kill()
                    process.wait(timeout=self.shutdown_timeout)
        finally:
            if self._log is not None:
                self._log.close()
                self._log = None


@contextmanager
def model_session(**kwargs: object) -> Iterator[ManagedModelServer]:
    with ManagedModelServer(**kwargs) as server:
        yield server
