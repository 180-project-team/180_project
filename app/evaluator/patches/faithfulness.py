from __future__ import annotations

from ragas.metrics.collections import Faithfulness


NEW_FAITHFULNESS_STATEMENT_INSTRUCTION = """
Given a question or an answer, split the answer into atomic statements.
Hard rules:
1) Split only what the answer explicitly says. Do not add facts from the question
or from retrieved knowledge.
2) Keep identifiers and qualifiers with the clause they belong to. Never copy an
identifier or qualifier from sentence A onto a fact from sentence B.
3) If the answer first says that something was not found or confirmed for one
source or entity, and later describes information from another source or entity,
keep these as SEPARATE statements. Do not merge them into one positive claim.
4) An absence claim must stay an absence claim. Do not turn it into a positive claim.
5) Do not resolve pronouns by substituting an identity that was not in that sentence.
If the sentence has no identifier, the statement must also have no identifier.
6) Do not infer that a source or entity contains or does not contain a fact unless
the answer states that about the same source or entity in the same clause.
7) Strip citation markers from statements. Do NOT copy "[1]", "[2]", "[n]",
"source [n]", or "источник [n]" into a statement. Keep the factual payload.
Those tags are answer formatting, not claims.
Return valid JSON only, matching the required schema.
""".strip()


NEW_FAITHFULNESS_NLI_INSTRUCTION = """
Judge whether each statement can be directly inferred from the given context.
Return verdict 1 only if the context entails the factual payload of the statement.
Return verdict 0 otherwise.
Hard rules:
1) Missing evidence is not proof. If the context does not mention an entity,
identifier, value, or event, you must NOT treat that as support for a statement
claiming that it does not exist or was not stated. Argument from ignorance => verdict 0.
2) Named entities and identifiers must match. A fact about entity Y does not
support the same fact attributed to entity X. An identifier from another source
does not support a statement about a specific different identifier.
3) A fact that is true of source or entity Y does not support the same fact
attributed to source or entity X.
4) If the statement has no identifier, you may use matching facts in context.
If the statement has a specific identifier, value, date, or number, the context
must contain that same detail for verdict 1.
5) Ignore citation tags. "[1]", "[2]", "source [n]", and "источник [n]" are
answer-side pointers. Context may not contain those labels. Do NOT return verdict 0
only because the tag is missing. Judge the remaining factual claim.
6) Speculation ("may be related", "possibly", "могут быть связаны") is not
entailed unless the context states that link. Verdict 0.
7) Do not use the user question. Use only the provided context.
Reason field:
- One short sentence, at most 25 words.
- State only why the verdict is 1 or 0.
- Do not debate both sides. Do not write however / nevertheless / if we assume.
- Do not quote these rules. Do not mention NLI.
Return valid JSON only, matching the required schema.
""".strip()


class FaithfulnessPatched(Faithfulness):
    """Faithfulness со старыми усиленными промптами без специфики прошлого проекта."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.statement_generator_prompt.instruction = (
            NEW_FAITHFULNESS_STATEMENT_INSTRUCTION
        )
        self.nli_statement_prompt.instruction = NEW_FAITHFULNESS_NLI_INSTRUCTION
