"""Deterministic example: replace with your agent adapter and LLM in production."""

from app.runner import CallableAdapter
from app.prompt_optimizer import PromptPatch, digest


class DemoAgent:
    def __init__(self, state):
        self.state = state

    def ask(self, question):
        # A deliberately weak target; only the prompt changes between runs.
        instruction = self.state.get("system.core").template
        if "verify" not in instruction.lower():
            return {"answer": "unknown", "trajectory": ["answered_without_verification"]}
        answers = {"2+2": "4", "1+1": "2", "3+3": "6", "4+4": "8"}
        return {"answer": answers.get(question, "unknown"), "trajectory": ["verified_answer"]}


def make_adapter(state, seed):
    return CallableAdapter(lambda: DemoAgent(state))


class DemoGenerator:
    def generate(self, state, lesson):
        component = state.get(lesson.component_id)
        return PromptPatch(
            component_id=lesson.component_id, operation="APPEND_RULE",
            before_hash=digest(component.template),
            payload="Verify the final answer against available information.",
            rule_id="verify_answer", rationale="Reduce incorrect answers by adding verification",
            evidence_refs=lesson.evidence_refs,
        )


def make_generator():
    return DemoGenerator()
