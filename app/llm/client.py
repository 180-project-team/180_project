"""Reusable OpenAI-compatible LLM client and optional local server lifecycle.

The orchestrator owns one LLMService and can share it with optimizers or other
modules. AgentRunner deliberately has no dependency on this service.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .server import ManagedModelServer


class LLMService:
    """Simple synchronous LLM API: start(), ask()/chat(), stop().

    `command` launches a local OpenAI-compatible server (llama.cpp, vLLM,
    etc.). Without `command`, only connects to an existing server. stop()
    never shuts down an external server. Use `with LLMService(...) as llm`.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str = "EMPTY",
        command: Sequence[str] | None = None,
        cwd: str | Path | None = None,
        startup_timeout: float = 120.0,
        request_timeout: float = 120.0,
        shutdown_timeout: float = 10.0,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        if request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if not (base_url.startswith("http://") or base_url.startswith("https://")):
            raise ValueError("base_url must be an HTTP(S) URL")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.request_timeout = request_timeout
        self._started = False
        self._server = ManagedModelServer(
            command=command,
            health_url=f"{self.base_url}/models",
            api_key=api_key,
            cwd=Path(cwd) if cwd is not None else None,
            startup_timeout=startup_timeout,
            shutdown_timeout=shutdown_timeout,
        )

    def __enter__(self) -> "LLMService":
        try:
            self.start()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    def start(self) -> None:
        if self._started:
            raise RuntimeError("LLMService has already been started")
        try:
            self._server.start()
            self._started = True
        except BaseException:
            self._server.stop()
            raise

    def stop(self) -> None:
        self._started = False
        self._server.stop()

    def chat_response(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Return the full OpenAI-compatible chat-completions response."""
        if not self._started:
            raise RuntimeError("Start LLMService before sending requests")
        if not messages or any(not isinstance(m, Mapping) or not isinstance(m.get("role"), str) or "content" not in m for m in messages):
            raise ValueError("messages must contain role and content")
        payload: dict[str, Any] = {"model": self.model, "messages": [dict(m) for m in messages]}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            if max_tokens < 1:
                raise ValueError("max_tokens must be positive")
            payload["max_tokens"] = max_tokens
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.request_timeout) as response:
                data = json.load(response)
        except HTTPError as exc:
            detail = exc.read(1000).decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("LLM response must be a JSON object")
        return data

    def chat(self, messages: Sequence[Mapping[str, Any]], **options: Any) -> str:
        """Return just the assistant's text response."""
        response = self.chat_response(messages, **options)
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Invalid chat-completions response") from exc
        if not isinstance(content, str):
            raise ValueError("LLM response content is not text")
        return content

    def ask(self, prompt: str, **options: Any) -> str:
        """Single-prompt shorthand for orchestrators and optimization modules."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be non-empty")
        return self.chat([{"role": "user", "content": prompt}], **options)
