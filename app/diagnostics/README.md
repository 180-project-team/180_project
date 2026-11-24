# Diagnostics

Diagnostics анализирует результаты Runner и оценки Evaluator по заданиям,
сохраняет наблюдаемые нарушения и предлагает направления проверки.
[Архитектура и границы MVP](../../docs/diagnostics.md) описаны отдельно.

## Быстрый старт

Запускайте примеры из корня репозитория. Для двух файловых примеров можно
использовать сохранённые в репозитории артефакты Evaluator.

### Диагностика без LLM

```python
from pathlib import Path

from app.diagnostics import diagnose_run

inputs = Path("outputs/evaluation/example")
batch = diagnose_run(
    run_results_path=inputs / "runner_results.jsonl",
    evaluation_results_path=inputs / "results.jsonl",
    output_dir=Path("outputs/diagnostics/example"),
    run_id="run-1",
    version_id="agent-v1",
)
print(batch.summary.analyzed_cases)
```

### Диагностика с общим LLMService

```python
from pathlib import Path

from app.diagnostics import DiagnosisBatchResult, diagnose_run
from app.llm import LLMService


def diagnose_with_llm(llm_service: LLMService) -> DiagnosisBatchResult:
    inputs = Path("outputs/evaluation/example")
    return diagnose_run(
        run_results_path=inputs / "runner_results.jsonl",
        evaluation_results_path=inputs / "results.jsonl",
        output_dir=Path("outputs/diagnostics/hybrid"),
        llm_service=llm_service,
        run_id="run-1",
        version_id="agent-v1",
    )
```

Вызовите `diagnose_with_llm(llm_service)` с готовым сервисом внутри контекста
`with LLMService(...) as llm_service` вызывающей стороны. Настройки подключения,
жизненный цикл сервиса, каталоги артефактов и идентификаторы в целевой
архитектуре определяет оркестратор. Когда сервис передан, запрос выполняется
только для задания, удовлетворяющего [условиям fallback](#llm-fallback).

## Публичный API

Все интерфейсы импортируются из `app.diagnostics`. `diagnose_run()` — основная
точка интеграции для будущего оркестратора. Он получает два входных пути,
возвращает объект пачки и публикует отчёты. `load_diagnostic_cases()` возвращает
объединённые записи для повторного анализа и восстановления свидетельств.

```python
diagnose_run(
    *,
    run_results_path: str | Path,
    evaluation_results_path: str | Path,
    output_dir: str | Path,
    llm_service: LLMService | None = None,
    run_id: str | None = None,
    version_id: str | None = None,
    quality_thresholds: QualityThresholds = QualityThresholds(),
) -> DiagnosisBatchResult

load_diagnostic_cases(
    run_results_path: str | Path,
    evaluation_results_path: str | Path,
) -> list[dict[str, Any]]
```

`diagnose()` принимает итератор `AgentRunner.run()` или другие совместимые
отображения, включая результат файлового join. Возвращает один отчёт на запись
в исходном порядке и сводку. Пустой итератор даёт пустую пачку. Входные данные
сохраняются без изменений. Опциональный нормализатор возвращает события со
ссылками на существующие поля записи, а валидатор возвращает `bool` для
непустого ответа при `status="ok"`.

```python
diagnose(
    results: Iterable[Mapping[str, Any]],
    *,
    run_id: str | None = None,
    version_id: str | None = None,
    normalizer: Callable[[Mapping[str, Any]], Sequence[TraceEvent]] = normalize_trace,
    answer_validator: Callable[[Any], bool] | None = None,
    quality_thresholds: QualityThresholds = QualityThresholds(),
    classifier: FailureClassifier | None = None,
) -> DiagnosisBatchResult

diagnose_case(
    case: str | Mapping[str, Any],
    *,
    quality_thresholds: QualityThresholds = QualityThresholds(),
    classifier: FailureClassifier | None = None,
) -> DiagnosisReport

LLMClassifier(llm_service: LLMService)
LLMClassifier.classify(
    self, state: Mapping[str, Any], candidates: tuple[FailureType, ...],
) -> Classification
```

`diagnose_case()` принимает JSON-строку или отображение. Требуются непустой
`question`, поле `answer` с допустимым `null`, `status` со значением `ok`,
`error` или `unknown`, массив `trajectory` либо `null` и объект `metrics`
(может быть пустым). `case_id` допускает строку, целое число или `null`.
`run_id`, `version_id` и `traceback` допускают строку или `null`.
Ошибка проверки входа приводит к `ValueError`.

В API для объектов `case_id` берётся из записи, с резервным полем `id`
при отсутствии ключа `case_id`. Явные аргументы `run_id` и `version_id`
имеют приоритет над полями записи. При отсутствии обоих источников
соответствующий идентификатор будет `null`. Ошибка обработки одной записи,
например исключение валидатора, даёт `DIAGNOSTIC_ERROR` без частичных сигналов.
Ошибки самого итератора и запись, которая не является отображением,
передаются вызывающей стороне.

`FailureClassifier` — протокол с методом `classify(state, candidates)`.
Для гибридного вызова `diagnose()` или `diagnose_case()` передайте
`classifier=LLMClassifier(llm_service)`. Для отдельных проверок доступны
`normalize_trace(result)` и `check_rules(result, events, answer_validator=...)`.
Исключение валидатора при прямом вызове `check_rules()` передаётся вызывающей стороне.

## Входные файлы и объединение

Оба входа — UTF-8 JSONL. Непустая строка содержит один JSON-объект. Пустые
строки пропускаются. Ниже показаны запись Runner и запись Evaluator для одного задания.

```jsonl
{"id":" 1 ","question":"Who directed the first film?","status":"ok","answer":"Alice","trajectory":[{"action":"bm25_search","args":{"query":"first film director"},"observation":"Alice directed the first film."}]}
```

```jsonl
{"id":1,"status":"ok","fact_recall":1.0,"fact_precision":1.0,"retrieval_fact_recall":1.0,"faithfulness":null,"duration_ms":80,"tool_calls_count":1,"successful_tool_calls_count":1,"failed_tool_calls_count":0}
```

- В обоих файлах обязательны `id` и `status` со значением `ok` или `error`.
  В Runner требуется непустой `question`. `trajectory` может отсутствовать,
  быть массивом или `null`. `answer`, `error` и остальные поля Runner
  сохраняются при наличии. `case_id`, `evaluation_status`, `metrics` и
  `execution_metrics` зарезервированы для объединения и в Runner недопустимы.
- Строковые и целочисленные ID приводятся к `str(id).strip()`. Пустые строки,
  логические значения, другие типы ID и дубликаты после нормализации приводят
  к `ValueError`. В каждом файле должна быть хотя бы одна запись, а наборы ID
  должны совпадать. При несовпадении ошибка перечисляет отсутствующие ID.
- Записи возвращаются в порядке Runner. Его `id` становится `case_id`.
  `status` сохраняет результат выполнения агента, `evaluation_status` —
  результат оценки из одноимённого поля `status` файла Evaluator.
- Оценки качества Evaluator из полей верхнего уровня попадают в `metrics`.
  В `execution_metrics` попадают доступные `duration_ms`, `tool_calls_count`,
  `successful_tool_calls_count` и `failed_tool_calls_count` Evaluator.
  Служебные значения переносятся без пересчёта и проверки диапазона.
  Одноимённые поля Runner остаются на верхнем уровне.

### Интерпретация оценок

| Ключ в `metrics` | Что оценивает Evaluator |
| --- | --- |
| `fact_recall` | Покрытие эталонных фактов финальным ответом |
| `fact_precision` | Соответствие утверждений финального ответа эталонным фактам |
| `retrieval_fact_recall` | Покрытие эталонных фактов успешными наблюдениями инструментов |
| `faithfulness` | Подтверждённость утверждений финального ответа успешными наблюдениями инструментов |

Отсутствующие ключи и `null` пропускаются при проверке качества. `0` —
измеренная оценка. Допустимы числа в `[0, 1]` без логических значений.
При успешном запуске и без ошибки оценки `diagnose()` отмечает строки,
логические значения и числа вне диапазона через `invalid_quality_metrics`.
`diagnose_case()` отклоняет такие значения через `ValueError`. Для прочих
ключей своего объекта `metrics` он допускает конечные числа и `null`.
Файловый загрузчик переносит поля как есть и отклоняет синтаксически
недопустимые константы JSON `NaN`, `Infinity` и `-Infinity`.

`QualityThresholds` задаёт для каждой из четырёх метрик отдельный
настраиваемый эвристический порог `0.8`. Низкой считается оценка строго
ниже своего порога. Ошибка оценки создаёт сигнал `evaluation_error` и
оставляет доступными проверки Runner. Ошибка отдельной метрики может дать
`null` при `evaluation_status="ok"`.

Низкие оценки дают наблюдение `low_quality_metrics` и гипотезы по сочетаниям
метрик. Например, низкий `fact_recall` при достаточном `retrieval_fact_recall`
предлагает проверить полноту ответа и интерпретацию инструментальных данных.
Высокий `retrieval_fact_recall` говорит о покрытии фактов в собранных наблюдениях.
Полноту фактического финального контекста по нему установить нельзя.

## LLM fallback

Условная семантическая проверка требует низкой оценки, подходящих кандидатов
из сигналов качества, непустого пригодного вопроса и ответа и хотя бы одного
содержательного успешного наблюдения. Установленное нарушение формата ответа
уровня `FAILURE` исключает вызов. Даже для одной предложенной категории
сохраняется вариант `UNKNOWN`.

`LLMClassifier` обращается к `LLMService.chat()` и проверяет JSON-объект с
единственным строковым полем `failure_type` из предложенного набора.
Запрос содержит оценки, вопрос и ответ до 2000 символов каждый, до восьми
сигналов, восьми гипотез и восьми выбранных шагов с исходными индексами.
Длина наблюдения ограничена 1200 символами, запроса поиска — 320.
Необработанные поля `raw_output` и `traceback` исключены, распознаваемые
учётные данные в тексте маскируются.

При принятой категории добавляется `classifier_selection` с низкой
уверенностью и ссылками на метрики и наблюдения. Если ведущим был общий
сигнал низкого качества, категория уточняет отчёт. Основное техническое
нарушение сохраняет приоритет. Ответ `UNKNOWN`, невалидный ответ и
исключение сохраняют локальный диагноз. В `classifier_attempt` фиксируются
`accepted`, `abstained` или `unavailable`. Без попытки это поле равно `null`.
`LLMClassifier` оставляет `cost_usd` неизвестной.

## Выходные файлы

`diagnose_run()` создаёт каталог вывода и сохраняет UTF-8 файлы.
`diagnoses.jsonl` содержит один `DiagnosisReport` на строку в порядке Runner.
`summary.json` содержит объект `DiagnosisSummary`. Возвращаемый
`DiagnosisBatchResult` содержит те же `reports` и `summary`. Перечисления
сериализуются строками, `None` — как `null`.

### DiagnosisReport

| Поля | Содержание |
| --- | --- |
| `case_id`, `run_id`, `version_id` | Идентификаторы задачи, запуска и версии |
| `status`, `failure_type` | Статус диагностики, основная категория или `UNKNOWN` |
| `critical_step`, `evidence_refs` | Возможный проблемный шаг либо `null`, ссылки на наблюдения всех сигналов без дублей |
| `rationale`, `confidence` | Обоснование вывода, уверенность `low`, `medium` или `high` без вероятностной калибровки |
| `suggested_scope` | Направление основной проверки, одно из `PROMPT`, `STRUCTURAL`, `CODE` и `UNKNOWN` |
| `signals` | Наблюдения правил и классификатора, включая предупреждения |
| `hypotheses` | Направления проверки из основного сигнала и подходящих предупреждений, с полями `hypothesis_id`, `scope`, `advice` |
| `classifier_attempt` | Статус попытки семантической классификации и доступная стоимость |

| Статус | Условие |
| --- | --- |
| `FAILURE_FOUND` | Есть сигнал уровня `FAILURE`. Итоговый ответ при этом может быть правильным |
| `NO_FAILURE_DETECTED` | Runner завершился успешно, Evaluator не сообщил ошибку, доступные проверки не нашли сигналов уровня `FAILURE`. Отсутствующие метрики остаются неизвестными |
| `INSUFFICIENT_DATA` | Сигналов уровня `FAILURE` нет, а успешное выполнение не подтверждено либо Evaluator сообщил об ошибке оценки |
| `DIAGNOSTIC_ERROR` | Исключение при диагностике отдельной записи |

Сигнал содержит `rule_id`, `failure_type`, `level`, `step_index`,
`evidence_refs`, `rationale`, `confidence` и опциональный `component_id`.
Основным обычно становится первый сигнал `FAILURE` с известной категорией.
При отсутствии такой категории берётся первый `FAILURE`. Принятый
fallback может уточнить основной сигнал низких метрик.

### Восстановление EvidenceRef

Сохраните оба исходных JSONL, которые использовались для диагностики.
Восстановите объединённую запись через `load_diagnostic_cases()`, выберите
её по нормализованному `case_id` отчёта и последовательно примените ключи
и индексы из `EvidenceRef.path`.

| Путь в объединённой записи | Исходное значение |
| --- | --- |
| `["trajectory", 0, "observation"]` | `trajectory[0].observation` в Runner |
| `["status"]`, `["error"]` | Одноимённые поля Runner |
| `["evaluation_status"]` | `status` в Evaluator |
| `["metrics", "fact_recall"]` | `fact_recall` в Evaluator |

Для прямых вызовов `diagnose()` и `diagnose_case()` путь отсчитывается
от переданного отображения. `step_index` и `critical_step` используют
исходные индексы `trajectory` с нуля. Локальный `event_id` выводится из
пути события. `tool_call_id` связывает вызов инструмента с результатом
в пределах записи. При отсутствующем ответе сигнал опирается на
существующий `status`, если ключа `answer` в записи нет.

### DiagnosisSummary и повторный запуск

`analyzed_cases` учитывает все отчёты. `failed_cases` считает только
`FAILURE_FOUND`, по одной основной категории на отчёт в `failure_counts`.
`report_indices` ссылается на позиции строк в `diagnoses.jsonl` с нуля.
`patterns` группирует сигналы с одинаковыми `rule_id`, `failure_type` и
`level`, встреченные хотя бы в двух разных отчётах с одинаковыми `run_id`
и `version_id`. Каждый паттерн хранит эти поля и свои `report_indices`.
В сводку идентификатор запуска или версии переносится, если соответствующее
значение одинаково у всех отчётов.

Повторный запуск в одном каталоге заменяет оба отчёта целиком. Сначала
проверяются входы и пересечения путей, затем оба файла записываются во
временный каталог. Замена атомарна для каждого файла отдельно. При ошибке
между заменами в каталоге может оказаться смешанная пара. Вызывающая сторона
получает исключение файловой системы и может повторить запуск. Для
параллельных запусков задавайте разные каталоги вывода.

## Тестирование и демонстрации

Команды выполняются из корня репозитория с установленным uv.

```shell
uv run --locked python -m unittest discover -s tests -v
uv run --locked python -m unittest discover -s tests -p "test_diagnostics*.py" -v
uv run --locked python -m unittest discover -s tests -p "test_diagnostics_integration.py" -k test_runner_evaluator_and_diagnostics_publish_consistent_reports -v
uv run --locked python -m examples.diagnose_runner
uv run --locked python -m examples.compare_skim_diagnostics tests/fixtures/skim_diagnostic_cases.jsonl
```

При заполненном кеше uv можно добавить `--offline` после `uv run`. Сквозной
тест использует реальные Runner, Evaluator и Diagnostics с тестовым агентом
и подставными ответами моделей. Демонстрация Runner печатает пять
синтетических сценариев и их диагностику. Сравнение вычисляет совпадения
с контрольной разметкой и долю `UNKNOWN`. Для сравнения с гибридным режимом
`compare(cases, classifier=...)` принимает готовый классификатор. Каждая запись
сравнения содержит объект `case` для `diagnose_case()`, ожидаемый
`failure_type` и опциональный `critical_step`. Эти проверки подтверждают
работу контрактов на контролируемых данных.
