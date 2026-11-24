"""Stable suggestions tied to observed failure types."""

from .models import DiagnosticSignal, FailureType, Hypothesis, SuggestedScope


_CATALOG = {
    FailureType.INVALID_TOOL_CALL: Hypothesis(
        "check_tool_contract", SuggestedScope.PROMPT,
        "Compare the emitted call with the tool schema and add a valid example for the failing argument.",
    ),
    FailureType.TOOL_EXECUTION_FAILURE: Hypothesis(
        "check_tool_runtime", SuggestedScope.STRUCTURAL,
        "Check tool availability and timeout settings. Verify whether a retry recovers the call.",
    ),
    FailureType.RETRIEVAL_FAILURE: Hypothesis(
        "inspect_supporting_document_recall", SuggestedScope.STRUCTURAL,
        "Verify the reference documents are indexed and compare their ranks under query and depth variants.",
    ),
    FailureType.CONTEXT_LOSS: Hypothesis(
        "inspect_context_selection", SuggestedScope.STRUCTURAL,
        "Compare retrieved evidence with the context sent to the final model and inspect truncation and filtering.",
    ),
    FailureType.INCOMPLETE_ANSWER: Hypothesis(
        "check_required_answer_facts", SuggestedScope.PROMPT,
        "Check whether the answer prompt asks the model to cover every required part of the question.",
    ),
    FailureType.TOOL_OUTPUT_MISINTERPRETATION: Hypothesis(
        "check_evidence_interpretation", SuggestedScope.PROMPT,
        "Compare the answer against the observed passages and review instructions for combining evidence.",
    ),
    FailureType.UNSUPPORTED_ANSWER: Hypothesis(
        "check_claim_grounding", SuggestedScope.PROMPT,
        "Check whether the final-answer instructions require each claim to be supported by the visible context.",
    ),
    FailureType.OUTPUT_FORMAT_ERROR: Hypothesis(
        "check_answer_format", SuggestedScope.PROMPT,
        "Compare the final response with the expected output format and check the answer extraction.",
    ),
}


def hypotheses_for(signal: DiagnosticSignal) -> list[Hypothesis]:
    if signal.failure_type == FailureType.PREMATURE_STOP and signal.rule_id == "budget_stop":
        return [Hypothesis(
            "review_execution_budget", SuggestedScope.STRUCTURAL,
            "Review the budget and the actions that consumed it before increasing the limit.",
        )]
    hypothesis = _CATALOG.get(signal.failure_type)
    return [hypothesis] if hypothesis else []
