from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any

from ragas.metrics.collections import ContextRecall

from .patches import FactualCorrectnessPatched, FaithfulnessPatched


LOGGER = logging.getLogger(__name__)
METRIC_NAMES = (
    "fact_recall",
    "fact_precision",
    "retrieval_fact_recall",
    "faithfulness",
)


def reference_facts_to_text(reference_facts: Sequence[str]) -> str:
    return "\n".join(
        f"{number}. {fact.strip()}"
        for number, fact in enumerate(reference_facts, start=1)
        if isinstance(fact, str) and fact.strip()
    )


def _metric_value(result: Any) -> float | None:
    value = getattr(result, "value", result)
    if value is None:
        return None

    score = float(value)
    return None if math.isnan(score) else score


class RagasMetrics:
    def __init__(self, llm: Any) -> None:
        self.fact_recall = FactualCorrectnessPatched(llm=llm, mode="recall")
        self.fact_precision = FactualCorrectnessPatched(llm=llm, mode="precision")
        self.retrieval_fact_recall = ContextRecall(llm=llm)
        self.faithfulness = FaithfulnessPatched(llm=llm)

    def calculate(
        self,
        *,
        question: str,
        answer: str,
        reference_facts: Sequence[str],
        contexts: list[str],
    ) -> dict[str, float | None]:
        values = {name: None for name in METRIC_NAMES}
        reference = reference_facts_to_text(reference_facts)

        calculations = (
            (
                "fact_recall",
                lambda: self.fact_recall.score(
                    response=answer,
                    reference=reference,
                ),
            ),
            (
                "fact_precision",
                lambda: self.fact_precision.score(
                    response=answer,
                    reference=reference,
                ),
            ),
            (
                "retrieval_fact_recall",
                lambda: self.retrieval_fact_recall.score(
                    user_input=question,
                    retrieved_contexts=contexts,
                    reference=reference,
                ),
            ),
            (
                "faithfulness",
                lambda: self.faithfulness.score(
                    user_input=question,
                    response=answer,
                    retrieved_contexts=contexts,
                ),
            ),
        )

        for name, calculation in calculations:
            try:
                values[name] = _metric_value(calculation())
            except Exception:
                LOGGER.exception("Не удалось посчитать метрику %s", name)
                values[name] = None

        return values
