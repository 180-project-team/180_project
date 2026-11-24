"""Immutable prompt versions and targeted, auditable prompt changes."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_hash(value: Any) -> str:
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


@dataclass(frozen=True)
class PromptComponent:
    component_id: str
    role: str
    template: str
    variables: tuple[str, ...] = ()
    editable: bool = True
    max_length: int = 20000

    def __post_init__(self) -> None:
        object.__setattr__(self, "variables", tuple(self.variables))
        if not self.component_id or not self.role or not isinstance(self.template, str):
            raise ValueError("PromptComponent requires component_id, role and string template")
        if self.max_length < 1 or len(self.template) > self.max_length:
            raise ValueError(f"Invalid max_length for {self.component_id}")
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("Duplicate declared template variables")

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id, "role": self.role,
            "template": self.template, "variables": list(self.variables),
            "editable": self.editable, "max_length": self.max_length,
        }


@dataclass(frozen=True)
class PromptState:
    components: tuple[PromptComponent, ...]
    parent_id: str | None = None
    renderer_version: str = "v1"
    metadata: tuple[tuple[str, str], ...] = ()
    prompt_version_id: str = field(init=False)

    def __post_init__(self) -> None:
        components = tuple(self.components)
        object.__setattr__(self, "components", components)
        metadata_items = self.metadata.items() if isinstance(self.metadata, Mapping) else self.metadata
        object.__setattr__(self, "metadata", tuple(sorted((str(k), str(v)) for k, v in metadata_items)))
        if not components or len({c.component_id for c in components}) != len(components):
            raise ValueError("PromptState requires unique, non-empty components")
        content = {
            "components": [c.to_dict() for c in components],
            "renderer_version": self.renderer_version,
        }
        object.__setattr__(self, "prompt_version_id", "ps-" + _json_hash(content)[:20])

    def get(self, component_id: str) -> PromptComponent:
        for component in self.components:
            if component.component_id == component_id:
                return component
        raise KeyError(f"Unknown prompt component: {component_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_version_id": self.prompt_version_id,
            "parent_id": self.parent_id,
            "renderer_version": self.renderer_version,
            "metadata": dict(self.metadata),
            "components": [c.to_dict() for c in self.components],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PromptState":
        state = cls(
            components=tuple(PromptComponent(**component) for component in data["components"]),
            parent_id=data.get("parent_id"),
            renderer_version=data.get("renderer_version", "v1"),
            metadata=tuple(data.get("metadata", {}).items()),
        )
        if data.get("prompt_version_id", state.prompt_version_id) != state.prompt_version_id:
            raise ValueError("PromptState content hash does not match stored version")
        return state

@dataclass(frozen=True)
class PromptLesson:
    pattern_id: str
    component_id: str
    instruction: str
    evidence_refs: tuple[str, ...]

@dataclass(frozen=True)
class PromptPatch:
    component_id: str
    operation: str
    before_hash: str
    payload: str
    rationale: str
    evidence_refs: tuple[str, ...] = ()
    rule_id: str | None = None
    patch_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch_id": self.patch_id, "component_id": self.component_id,
            "operation": self.operation, "before_hash": self.before_hash,
            "payload": self.payload, "rationale": self.rationale,
            "evidence_refs": list(self.evidence_refs), "rule_id": self.rule_id,
        }
