"""Batch counts and repeated rule observations, without semantic clustering."""

from collections import Counter, defaultdict
from typing import Sequence

from .models import (
    BehaviorPattern, DiagnosisReport, DiagnosisStatus, DiagnosisSummary,
    FailureType, SignalLevel,
)


def summarize(reports: Sequence[DiagnosisReport]) -> DiagnosisSummary:
    """Aggregate all cases, separating rule patterns by source identifiers.

    Summary identifiers are populated only when shared by every report.
    In mixed batches the report indices preserve the source of each count.
    """
    runs = {report.run_id for report in reports}
    versions = {report.version_id for report in reports}
    counts = Counter(report.failure_type for report in reports
                     if report.status == DiagnosisStatus.FAILURE_FOUND)
    groups: dict[tuple[str | None, str | None, str, FailureType, SignalLevel], list[int]] = defaultdict(list)
    for index, report in enumerate(reports):
        keys = dict.fromkeys((report.run_id, report.version_id, signal.rule_id,
                              signal.failure_type, signal.level) for signal in report.signals)
        for key in keys:
            groups[key].append(index)
    patterns = [
        BehaviorPattern(rule, category, level, run, version, indices)
        for (run, version, rule, category, level), indices in groups.items()
        if len(indices) > 1
    ]
    return DiagnosisSummary(
        analyzed_cases=len(reports), failed_cases=sum(counts.values()),
        failure_counts=dict(counts), report_indices=list(range(len(reports))),
        patterns=patterns,
        run_id=next(iter(runs)) if len(runs) == 1 else None,
        version_id=next(iter(versions)) if len(versions) == 1 else None,
    )
