"""Compare diagnostic categories and steps with labeled Runner cases."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.diagnostics import (
    DiagnosisReport, DiagnosisStatus, FailureClassifier, FailureType,
    diagnose_case,
)


@dataclass
class Score:
    cases: int = 0
    correct_categories: int = 0
    unknown: int = 0
    failures: int = 0
    unattributed_failures: int = 0
    labeled_steps: int = 0
    correct_steps: int = 0
    classifier_calls: int = 0
    reported_cost_cases: int = 0
    reported_cost_usd: float = 0

    def record(
        self, report: DiagnosisReport, label: FailureType, step: int | None,
    ) -> None:
        self.cases += 1
        self.correct_categories += report.failure_type == label
        self.unknown += report.failure_type == FailureType.UNKNOWN
        if report.status == DiagnosisStatus.FAILURE_FOUND:
            self.failures += 1
            self.unattributed_failures += report.failure_type == FailureType.UNKNOWN
        if step is not None:
            self.labeled_steps += 1
            self.correct_steps += report.critical_step == step
        if report.classifier_attempt is not None:
            self.classifier_calls += 1
            cost = report.classifier_attempt.cost_usd
            if cost is not None:
                self.reported_cost_cases += 1
                self.reported_cost_usd += cost

    def summary(self) -> dict:
        return {
            "cases": self.cases,
            "correct_categories": self.correct_categories,
            "category_accuracy": self.correct_categories / self.cases if self.cases else None,
            "unknown_rate": self.unknown / self.cases if self.cases else None,
            "unattributed_failure_rate": self.unattributed_failures / self.failures
            if self.failures else None,
            "labeled_steps": self.labeled_steps,
            "step_accuracy": self.correct_steps / self.labeled_steps if self.labeled_steps else None,
            "classifier_calls": self.classifier_calls,
            "classifier_call_rate": self.classifier_calls / self.cases if self.cases else None,
            "reported_cost_cases": self.reported_cost_cases,
            "reported_cost_usd": self.reported_cost_usd if self.reported_cost_cases else None,
            "mean_cost_per_case_usd": self.reported_cost_usd / self.cases
            if self.cases and self.reported_cost_cases == self.classifier_calls
            else None,
        }


def compare(
    cases: Iterable[dict], classifier: FailureClassifier | None = None,
) -> dict:
    rules = Score()
    hybrid = Score()
    for labeled in cases:
        case = labeled["case"]
        label = FailureType(labeled["failure_type"])
        step = labeled.get("critical_step")
        rules.record(diagnose_case(case), label, step)
        if classifier is not None:
            hybrid.record(diagnose_case(case, classifier=classifier), label, step)
    result = {"rules": rules.summary()}
    if classifier is not None:
        result["hybrid"] = hybrid.summary()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path, help="JSONL file with labeled Runner cases and Evaluator scores")
    args = parser.parse_args()
    with args.cases.open(encoding="utf-8") as source:
        result = compare(json.loads(line) for line in source if line.strip())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
