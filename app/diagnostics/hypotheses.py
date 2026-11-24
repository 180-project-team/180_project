"""Stable suggestions tied to observed failure types."""

from .models import DiagnosticSignal, FailureType, Hypothesis, SignalLevel, SuggestedScope


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
        "Inspect tool observations for the required facts and review the search queries and retrieval depth.",
    ),
    FailureType.CONTEXT_LOSS: Hypothesis(
        "inspect_context_selection", SuggestedScope.STRUCTURAL,
        "If the final model input is available, compare it with retrieved evidence and inspect truncation and filtering.",
    ),
    FailureType.INCOMPLETE_ANSWER: Hypothesis(
        "check_required_answer_facts", SuggestedScope.PROMPT,
        "Identify which required parts of the question the answer omits and check the answer instructions.",
    ),
    FailureType.TOOL_OUTPUT_MISINTERPRETATION: Hypothesis(
        "check_evidence_interpretation", SuggestedScope.PROMPT,
        "Compare the answer against the observed passages and review instructions for combining evidence.",
    ),
    FailureType.UNSUPPORTED_ANSWER: Hypothesis(
        "check_claim_grounding", SuggestedScope.PROMPT,
        "Compare claims in the answer with the tool observations and identify any unsupported claims.",
    ),
    FailureType.OUTPUT_FORMAT_ERROR: Hypothesis(
        "check_answer_format", SuggestedScope.PROMPT,
        "Compare the final response with the expected output format and check the answer extraction.",
    ),
}

_CHECK_ANSWER_RELEVANCE = Hypothesis(
    "check_answer_relevance", SuggestedScope.PROMPT,
    "Check whether the extra details address the question and are consistent with the reference facts.",
)
_QUALITY_HYPOTHESES = {
    "possible_retrieval_failure": (_CATALOG[FailureType.RETRIEVAL_FAILURE],),
    "low_fact_recall": (_CATALOG[FailureType.INCOMPLETE_ANSWER],),
    "answer_coverage_gap": (
        _CATALOG[FailureType.INCOMPLETE_ANSWER],
        _CATALOG[FailureType.TOOL_OUTPUT_MISINTERPRETATION],
        _CATALOG[FailureType.CONTEXT_LOSS],
    ),
    "low_fact_precision": (_CHECK_ANSWER_RELEVANCE,),
    "supported_precision_gap": (_CHECK_ANSWER_RELEVANCE,),
    "unsupported_precision_gap": (
        _CATALOG[FailureType.UNSUPPORTED_ANSWER],
        Hypothesis("check_claim_correctness", SuggestedScope.PROMPT,
                   "Compare disputed claims with the reference facts and the observed tool evidence."),
    ),
    "low_faithfulness": (_CATALOG[FailureType.UNSUPPORTED_ANSWER],),
}


def hypotheses_for(signal: DiagnosticSignal) -> list[Hypothesis]:
    if signal.level == SignalLevel.WARNING and signal.rule_id != "classifier_selection":
        return list(_QUALITY_HYPOTHESES.get(signal.rule_id, ()))
    if signal.failure_type == FailureType.PREMATURE_STOP and signal.rule_id == "budget_stop":
        return [Hypothesis(
            "review_execution_budget", SuggestedScope.STRUCTURAL,
            "Review the budget and the actions that consumed it before increasing the limit.",
        )]
    hypothesis = _CATALOG.get(signal.failure_type)
    return [hypothesis] if hypothesis else []
