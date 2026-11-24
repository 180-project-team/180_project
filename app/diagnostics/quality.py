"""Interpret Evaluator scores as observations about answer and retrieval quality."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .models import Confidence, DiagnosticSignal, EvidenceRef, FailureType, SignalLevel


QUALITY_METRICS = ("fact_recall", "fact_precision", "retrieval_fact_recall", "faithfulness")


def is_quality_score(value: Any) -> bool:
    return type(value) in (int, float) and 0 <= value <= 1


@dataclass(frozen=True)
class QualityThresholds:
    """Heuristic lower bounds for the four Evaluator quality scores."""

    fact_recall: float = 0.8
    fact_precision: float = 0.8
    retrieval_fact_recall: float = 0.8
    faithfulness: float = 0.8

    def __post_init__(self) -> None:
        for name in QUALITY_METRICS:
            if not is_quality_score(getattr(self, name)):
                raise ValueError(f"Quality threshold {name!r} must be a number in [0, 1]")


def _quality_warning(rule_id: str, rationale: str, *metrics: str) -> DiagnosticSignal:
    return DiagnosticSignal(
        rule_id, FailureType.UNKNOWN, SignalLevel.WARNING, None,
        [EvidenceRef(["metrics", name]) for name in metrics], rationale, Confidence.LOW,
    )


def quality_signals(
    row: Mapping[str, Any], thresholds: QualityThresholds,
) -> list[DiagnosticSignal]:
    if row.get("evaluation_status") == "error":
        return [DiagnosticSignal(
            "evaluation_error", FailureType.UNKNOWN, SignalLevel.WARNING, None,
            [EvidenceRef(["evaluation_status"])], "Evaluator could not assess this result.", Confidence.HIGH,
        )]
    if row.get("status") != "ok":
        return []
    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping):
        return []

    scores = {}
    invalid = []
    for name in QUALITY_METRICS:
        if name not in metrics or metrics[name] is None:
            continue
        value = metrics[name]
        if is_quality_score(value):
            scores[name] = value
        else:
            invalid.append(name)

    signals = []
    low = [name for name, value in scores.items() if value < getattr(thresholds, name)]
    if low:
        signals.append(DiagnosticSignal(
            "low_quality_metrics", FailureType.UNKNOWN, SignalLevel.FAILURE, None,
            [EvidenceRef(["metrics", name]) for name in low],
            "One or more scores fall below their thresholds. The cause requires more evidence.",
            Confidence.LOW,
        ))
    if invalid:
        signals.append(DiagnosticSignal(
            "invalid_quality_metrics", FailureType.UNKNOWN, SignalLevel.WARNING, None,
            [EvidenceRef(["metrics", name]) for name in invalid],
            "Evaluator quality scores must be numbers in [0, 1].", Confidence.HIGH,
        ))
    if "retrieval_fact_recall" in low:
        evidence = [EvidenceRef(["metrics", "retrieval_fact_recall"])]
        if "fact_recall" in low:
            evidence.append(EvidenceRef(["metrics", "fact_recall"]))
        signals.append(DiagnosticSignal(
            "possible_retrieval_failure", FailureType.RETRIEVAL_FAILURE, SignalLevel.WARNING,
            None, evidence, "Successful tool observations cover too few reference facts.", Confidence.LOW,
        ))
    if "fact_recall" in low:
        if "retrieval_fact_recall" in scores and "retrieval_fact_recall" not in low:
            signals.append(_quality_warning(
                "answer_coverage_gap",
                "Tool observations meet the retrieval threshold while the answer misses reference facts. "
                "The final model context has not been measured.",
                "fact_recall", "retrieval_fact_recall",
            ))
        else:
            signals.append(_quality_warning("low_fact_recall", "The answer misses reference facts.", "fact_recall"))
    if "faithfulness" in low:
        signals.append(_quality_warning(
            "low_faithfulness", "Some answer claims have insufficient support in tool observations.", "faithfulness",
        ))
    if "fact_precision" in low:
        if "faithfulness" in low:
            signals.append(_quality_warning(
                "unsupported_precision_gap",
                "Some claims may be unsupported by tool observations or disagree with reference facts.",
                "fact_precision", "faithfulness",
            ))
        elif "faithfulness" in scores:
            signals.append(_quality_warning(
                "supported_precision_gap",
                "Some claims may be supported by tool observations and still add details outside the reference facts.",
                "fact_precision", "faithfulness",
            ))
        else:
            signals.append(_quality_warning(
                "low_fact_precision", "The answer has low agreement with the reference facts.", "fact_precision",
            ))
    return signals
