"""LLM-backed patch proposals using the project's shared LLMService."""

from __future__ import annotations

import json
from typing import Any, Protocol

from app.prompt_optimizer.models import PromptLesson, PromptPatch, PromptState, digest


class PromptModelPort(Protocol):
    def generate(self, state: PromptState, lesson: PromptLesson) -> PromptPatch: ...


class LLMPatchGenerator:
    """LLMService can be injected by an orchestrator; no model is started here."""

    def __init__(self, llm: Any) -> None:
        if not callable(getattr(llm, "chat", None)):
            raise TypeError("llm must provide chat(messages, **options)")
        self.llm = llm

    def generate(self, state: PromptState, lesson: PromptLesson) -> PromptPatch:
        component = state.get(lesson.component_id)
        system = (
            "You propose SMALL, reusable prompt changes for an agent. "
            "Never include benchmark answers, task identifiers, secrets or unrelated changes. "
            "Do not change template placeholders. Reply as one JSON object only: "
            '{"operation":"APPEND_RULE","payload":"...","rule_id":"...","rationale":"..."}. '
            "Alternative operation: REPLACE_COMPONENT (payload must contain the full revised component). "
            "Any rule_id must consist only of letters, numbers, underscore, dash or dot."
        )
        user = json.dumps({
            "component_id": component.component_id,
            "template": component.template,
            "lesson": lesson.instruction,
            "pattern": lesson.pattern_id,
            "max_length": component.max_length,
        }, ensure_ascii=False)
        raw = self.llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], temperature=0)
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        try:
            proposal = json.loads(text)
        except (ValueError, IndexError) as exc:
            raise ValueError("LLM did not return a valid JSON patch") from exc
        if not isinstance(proposal, dict) or not all(isinstance(proposal.get(key), str) for key in ("operation", "payload", "rationale")):
            raise ValueError("LLM patch is missing required string fields")
        return PromptPatch(
            component_id=lesson.component_id,
            operation=proposal["operation"],
            before_hash=digest(component.template),
            payload=proposal["payload"],
            rationale=proposal["rationale"],
            evidence_refs=lesson.evidence_refs,
            rule_id=proposal.get("rule_id"),
        )
