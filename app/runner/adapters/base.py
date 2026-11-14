"""Stable adapter protocol for local agents."""

from __future__ import annotations

from typing import Any, Callable, Protocol


class AgentAdapter(Protocol):
    """Agent-specific logic lives here; runner only calls these methods."""

    def start(self) -> None: ...
    def ask(self, question: str) -> Any: ...
    def stop(self) -> None: ...


class CallableAdapter:
    """Wrap a local Python object with ask(question) and optional close/stop.

    Only suitable for agents that release their own resources in stop/close.
    For guaranteed OS-process cleanup (incl. GPU memory), prefer a subprocess.
    """

    def __init__(self, factory: Callable[[], Any]) -> None:
        self.factory = factory
        self.agent: Any = None

    def start(self) -> None:
        self.agent = self.factory()
        start = getattr(self.agent, "start", None)
        if callable(start):
            start()

    def ask(self, question: str) -> Any:
        if self.agent is None:
            raise RuntimeError("Agent is not started")
        return self.agent.ask(question)

    def stop(self) -> None:
        agent, self.agent = self.agent, None
        if agent is not None:
            cleanup = getattr(agent, "stop", None)
            if not callable(cleanup):
                cleanup = getattr(agent, "close", None)
            if callable(cleanup):
                cleanup()
