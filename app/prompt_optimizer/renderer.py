"""Deterministic rendering and template-variable validation."""

from __future__ import annotations

import json
import string
from dataclasses import dataclass
from typing import Any, Mapping

from app.prompt_optimizer.models import PromptState, digest


def placeholders(template: str) -> set[str]:
    names: set[str] = set()
    for _, name, _, _ in string.Formatter().parse(template):
        if name is None:
            continue
        if not name.isidentifier():
            raise ValueError(f"Unsupported placeholder: {name!r}")
        names.add(name)
    return names


@dataclass(frozen=True)
class RenderedPromptBundle:
    prompt_version_id: str
    messages: tuple[dict[str, str], ...]
    tool_descriptions: dict[str, str]
    content_hash: str


class PromptRenderer:
    def validate(self, state: PromptState) -> None:
        for component in state.components:
            declared = set(component.variables)
            actual = placeholders(component.template)
            if actual != declared:
                raise ValueError(
                    f"Template variables for {component.component_id}: "
                    f"declared {sorted(declared)}, found {sorted(actual)}"
                )

    def render(self, state: PromptState, variables: Mapping[str, Any] | None = None) -> RenderedPromptBundle:
        self.validate(state)
        context = dict(variables or {})
        messages: list[dict[str, str]] = []
        tools: dict[str, str] = {}
        for component in state.components:
            missing = set(component.variables) - context.keys()
            if missing:
                raise ValueError(f"Missing values for {component.component_id}: {sorted(missing)}")
            content = component.template.format_map(context)
            if component.component_id.startswith("tools.") and component.component_id.endswith(".description"):
                tool_name = component.component_id[len("tools."):-len(".description")]
                tools[tool_name] = content
            else:
                role = component.role if component.role in {"system", "developer", "user"} else "system"
                messages.append({"role": role, "content": content, "component_id": component.component_id})
        serialized = json.dumps({"messages": messages, "tools": tools}, sort_keys=True, ensure_ascii=False)
        return RenderedPromptBundle(state.prompt_version_id, tuple(messages), tools, digest(serialized))

    def validate_bindable(self, state: PromptState) -> None:
        variables = {name: f"test_{name}" for c in state.components for name in c.variables}
        self.render(state, variables)
