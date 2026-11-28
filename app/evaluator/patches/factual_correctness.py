from __future__ import annotations

from ragas.metrics.collections import FactualCorrectness
from ragas.metrics.result import MetricResult


NEW_FC_CLAIM_DECOMPOSITION_INSTRUCTION = """
Decompose the input text into atomic, independently verifiable claims.

Strict constraints:
1) Do NOT add entities, organizations, products, roles, qualifiers, or facts
   that are not explicitly present in the input text.
2) Preserve subject wording as close to the source text as possible.
3) Do NOT infer background context.
4) Do NOT generalize or specialize facts from the source text.
5) Return only claims explicitly stated in the input text.

Output format:
Return valid JSON only, matching the required schema.
""".strip()


class FactualCorrectnessPatched(FactualCorrectness):
    """Factual Correctness с нашим промптом и исправленным recall."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt.instruction = NEW_FC_CLAIM_DECOMPOSITION_INSTRUCTION

    async def ascore(self, response: str, reference: str) -> MetricResult:
        if self.mode != "recall":
            return await super().ascore(response=response, reference=reference)

        if not response:
            raise ValueError("response is missing")
        if not reference:
            raise ValueError("reference is missing")

        # В RAGAS 0.4.4 recall использует true positives из расчёта precision.
        # Для recall проверяем именно факты из reference относительно ответа агента.
        reference_in_response = await self._decompose_and_verify_claims(
            reference,
            response,
        )

        if len(reference_in_response) == 0:
            return MetricResult(value=0.0)

        score = sum(bool(value) for value in reference_in_response) / len(
            reference_in_response
        )
        return MetricResult(value=round(float(score), 2))
