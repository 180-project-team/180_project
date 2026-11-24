"""Use case metrics and trace evidence to distinguish quality failure modes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .models import Confidence, DiagnosticSignal, EvidenceRef, FailureType, SignalLevel


@dataclass(frozen=True)
class QualityThresholds:
    answer_f1: float = 0.8
    fact_recall: float = 0.8
    retrieval_fact_recall: float = 0.8
    faithfulness: float = 0.8

    def __post_init__(self) -> None:
        for value in (self.answer_f1, self.fact_recall,
                      self.retrieval_fact_recall, self.faithfulness):
            if type(value) not in (int, float) or not 0 <= value <= 1:
                raise ValueError("Quality thresholds must be numbers in [0, 1]")


def _score(metrics: Mapping[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    return value if type(value) in (int, float) and 0 <= value <= 1 else None


def quality_signals(
    row: Mapping[str, Any], thresholds: QualityThresholds,
) -> list[DiagnosticSignal]:
    if row.get("status") != "ok":
        return []
    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping):
        return []

    fact_recall = _score(metrics, "fact_recall")
    retrieval_recall = _score(metrics, "retrieval_fact_recall")
    final_recall = _score(metrics, "final_context_fact_recall")
    faithfulness = _score(metrics, "faithfulness")
    low_metrics = [
        name for name, threshold in (
            ("fact_recall", thresholds.fact_recall), ("answer_f1", thresholds.answer_f1),
            ("answer_em", 1), ("retrieval_fact_recall", thresholds.retrieval_fact_recall),
            ("faithfulness", thresholds.faithfulness),
            ("final_context_fact_recall", thresholds.fact_recall),
        )
        if (value := _score(metrics, name)) is not None and value < threshold
    ]
    if not low_metrics:
        return []
    refs = [EvidenceRef(["metrics", name]) for name in low_metrics]
    if retrieval_recall == 1 and final_recall is not None and final_recall < 1:
        return [DiagnosticSignal(
            "measured_context_loss", FailureType.CONTEXT_LOSS, SignalLevel.FAILURE, None,
            refs + [EvidenceRef(["metrics", name]) for name in (
                "retrieval_fact_recall", "final_context_fact_recall"
            )],
            "The final model context covered fewer reference facts than the retrieved evidence.",
            Confidence.HIGH,
        )]
    if final_recall == 1 and fact_recall is not None and fact_recall < thresholds.fact_recall:
        return [DiagnosticSignal(
            "measured_answer_omission", FailureType.INCOMPLETE_ANSWER, SignalLevel.FAILURE, None,
            refs + [EvidenceRef(["metrics", "final_context_fact_recall"])],
            "The final model context covered all reference facts and the answer omitted some.",
            Confidence.HIGH,
        )]
    if (isinstance(row.get("final_context"), str) and row["final_context"].strip()
            and faithfulness is not None and faithfulness < thresholds.faithfulness):
        return [DiagnosticSignal(
            "unsupported_answer", FailureType.UNSUPPORTED_ANSWER, SignalLevel.FAILURE, None,
            refs + [EvidenceRef(["final_context"])],
            "The evaluator found insufficient support for the answer in the final context.",
            Confidence.MEDIUM,
        )]

    signals = [DiagnosticSignal(
        "low_quality_metrics", FailureType.UNKNOWN, SignalLevel.FAILURE, None, refs,
        "One or more case metrics are below threshold; attribution requires more evidence.",
        Confidence.HIGH,
    )]
    if retrieval_recall is not None and retrieval_recall < thresholds.retrieval_fact_recall:
        signals.append(DiagnosticSignal(
            "possible_retrieval_failure", FailureType.RETRIEVAL_FAILURE, SignalLevel.WARNING,
            None, [EvidenceRef(["metrics", "retrieval_fact_recall"])],
            "The retrieval metric is low; the trace does not establish a search failure.",
            Confidence.LOW,
        ))
    return signals
