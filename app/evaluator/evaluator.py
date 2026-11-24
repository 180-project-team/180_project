from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from app.llm import LLMService

from .metrics import METRIC_NAMES, RagasMetrics
from .ragas_llm import create_ragas_llm
from .report import write_evaluation_report
from .trajectory import (
    count_failed_tool_calls,
    count_successful_tool_calls,
    count_tool_calls,
    extract_tool_contexts,
    has_trajectory,
)


def _answer_as_text(answer: Any) -> str:
    if isinstance(answer, str):
        return answer.strip()
    if isinstance(answer, Sequence) and not isinstance(
        answer, (str, bytes, bytearray)
    ):
        return "\n".join(str(item) for item in answer if item is not None).strip()
    return "" if answer is None else str(answer).strip()


def _index_evaluation_dataset(
    evaluation_dataset: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}

    for position, item in enumerate(evaluation_dataset, start=1):
        if not isinstance(item, Mapping):
            raise TypeError(f"Строка #{position} в evaluation dataset должна быть объектом")

        item_id = item.get("id")
        if item_id is None or not str(item_id).strip():
            raise ValueError(f"В строке #{position} нет непустого id")
        item_id = str(item_id).strip()

        if item_id in indexed:
            raise ValueError(f"Повторяющийся id в evaluation dataset: {item_id}")

        question = item.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"У вопроса {item_id} нет непустого question")

        reference_answer = item.get("reference_answer")
        if not isinstance(reference_answer, str) or not reference_answer.strip():
            raise ValueError(f"У вопроса {item_id} нет непустого reference_answer")

        reference_facts = item.get("reference_facts")
        if (
            not isinstance(reference_facts, Sequence)
            or isinstance(reference_facts, (str, bytes, bytearray))
            or not reference_facts
        ):
            raise ValueError(
                f"У вопроса {item_id} reference_facts должен быть непустым списком"
            )

        if not all(isinstance(fact, str) and fact.strip() for fact in reference_facts):
            raise ValueError(
                f"У вопроса {item_id} в reference_facts есть пустой или некорректный факт"
            )

        indexed[item_id] = {
            **dict(item),
            "id": item_id,
            "question": question.strip(),
            "reference_answer": reference_answer.strip(),
            "reference_facts": [fact.strip() for fact in reference_facts],
        }

    return indexed


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Некорректный JSON в {path}, строка {line_number}"
                ) from exc

            if not isinstance(row, dict):
                raise ValueError(
                    f"Строка {line_number} в {path} должна содержать JSON-объект"
                )
            yield row


class Evaluator:
    def __init__(self, llm_service: LLMService) -> None:
        ragas_llm = create_ragas_llm(llm_service)
        self._metrics = RagasMetrics(ragas_llm)

    def evaluate(
        self,
        *,
        evaluation_dataset: Iterable[Mapping[str, Any]],
        run_results_path: str | Path,
        output_dir: str | Path,
    ) -> None:
        output_path = Path(output_dir)
        if not output_path.exists() or not output_path.is_dir():
            raise ValueError("output_dir должен указывать на существующую папку")

        results_path = Path(run_results_path)
        if not results_path.is_file():
            raise FileNotFoundError(results_path)

        dataset_by_id = _index_evaluation_dataset(evaluation_dataset)
        seen_ids: set[str] = set()
        evaluation_rows: list[dict[str, Any]] = []

        for run_result in _read_jsonl(results_path):
            result_id = run_result.get("id")
            if result_id is None or not str(result_id).strip():
                raise ValueError("В результате Runner нет непустого id")
            result_id = str(result_id).strip()

            if result_id in seen_ids:
                raise ValueError(f"Повторяющийся id в результатах Runner: {result_id}")
            seen_ids.add(result_id)

            if result_id not in dataset_by_id:
                raise ValueError(
                    f"В evaluation dataset нет разметки для результата {result_id}"
                )

            question = run_result.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"В результате Runner {result_id} нет непустого question")
            question = question.strip()

            annotation = dataset_by_id[result_id]
            if question != annotation["question"]:
                raise ValueError(
                    f"Question для {result_id} не совпадает в evaluation dataset и результатах Runner"
                )

            answer = _answer_as_text(run_result.get("answer"))
            trajectory = run_result.get("trajectory")
            contexts = extract_tool_contexts(trajectory)

            runner_status = str(run_result.get("status") or "").strip().lower()
            can_evaluate = runner_status == "ok" and bool(answer) and has_trajectory(
                trajectory
            )
            status = "ok" if can_evaluate else "error"

            metric_values = {name: None for name in METRIC_NAMES}
            if can_evaluate:
                metric_values = self._metrics.calculate(
                    question=question,
                    answer=answer,
                    reference_facts=annotation["reference_facts"],
                    contexts=contexts,
                )

            evaluation_rows.append(
                {
                    "id": result_id,
                    "status": status,
                    **metric_values,
                    "duration_ms": run_result.get("duration_ms"),
                    "tool_calls_count": count_tool_calls(trajectory),
                    "successful_tool_calls_count": count_successful_tool_calls(
                        trajectory
                    ),
                    "failed_tool_calls_count": count_failed_tool_calls(trajectory),
                }
            )

        write_evaluation_report(output_path, evaluation_rows)
