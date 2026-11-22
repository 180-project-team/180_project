from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


# Временно работаем с форматом trajectory от SkimSearchAgent.
# Потом заменим на общий формат из Runner.
_NON_TOOL_ACTIONS = {
    "answer",
    "submit",
    "stop",
    "budget",
    "fix",
    "fix_rejected",
    "none",
    "",
}


def _trajectory_steps(trajectory: Any) -> list[Mapping[str, Any]]:
    if not isinstance(trajectory, Sequence) or isinstance(
        trajectory, (str, bytes, bytearray)
    ):
        return []

    return [step for step in trajectory if isinstance(step, Mapping)]


def has_trajectory(trajectory: Any) -> bool:
    return bool(_trajectory_steps(trajectory))


def _is_tool_step(step: Mapping[str, Any]) -> bool:
    action = str(step.get("action") or "").strip().lower()
    observation = str(step.get("observation") or "").strip()

    if action in _NON_TOOL_ACTIONS:
        return False
    if observation == "(episode ended)":
        return False
    return True


def _is_failed_tool_step(step: Mapping[str, Any]) -> bool:
    observation = str(step.get("observation") or "").lstrip()
    return observation.startswith("ERROR")


def extract_tool_contexts(trajectory: Any) -> list[str]:
    contexts: list[str] = []

    for step in _trajectory_steps(trajectory):
        if not _is_tool_step(step) or _is_failed_tool_step(step):
            continue

        observation = step.get("observation")
        if isinstance(observation, str) and observation.strip():
            contexts.append(observation.strip())

    return contexts


def count_tool_calls(trajectory: Any) -> int:
    return sum(_is_tool_step(step) for step in _trajectory_steps(trajectory))


def count_successful_tool_calls(trajectory: Any) -> int:
    return sum(
        _is_tool_step(step) and not _is_failed_tool_step(step)
        for step in _trajectory_steps(trajectory)
    )


def count_failed_tool_calls(trajectory: Any) -> int:
    return sum(
        _is_tool_step(step) and _is_failed_tool_step(step)
        for step in _trajectory_steps(trajectory)
    )
