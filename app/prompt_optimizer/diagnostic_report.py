"""Read existing Diagnostics reports without running diagnostics again."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.diagnostics.models import DiagnosisStatus, SuggestedScope


def prompt_hypotheses(
    report: Mapping[str, Any], *, primary_scope_only: bool = False,
) -> tuple[Mapping[str, Any], ...]:
    """Select prompt advice, including secondary hypotheses when enabled."""
    if report.get("status") in ("DIAGNOSTIC_ERROR", "INSUFFICIENT_DATA"):
        return ()
    if primary_scope_only and report.get("suggested_scope") != "PROMPT":
        return ()
    if not report.get("evidence_refs"):
        return ()
    return tuple(
        hypothesis for hypothesis in report.get("hypotheses", [])
        if isinstance(hypothesis, Mapping)
        and hypothesis.get("scope") == "PROMPT"
        and isinstance(hypothesis.get("advice"), str)
        and hypothesis["advice"].strip()
    )


def _validate_report(value: Any, location: str) -> dict[str, Any]:
    required = {"status", "failure_type", "suggested_scope", "hypotheses", "evidence_refs"}
    if not isinstance(value, dict) or not required.issubset(value):
        raise ValueError(
            f"{location}: expected a DiagnosisReport; use diagnoses.jsonl, not summary.json"
        )
    if not isinstance(value["status"], str) or value["status"] not in DiagnosisStatus:
        raise ValueError(f"{location}: invalid diagnosis status")
    if not isinstance(value["suggested_scope"], str) or value["suggested_scope"] not in SuggestedScope:
        raise ValueError(f"{location}: invalid suggested_scope")
    if not isinstance(value["failure_type"], str):
        raise ValueError(f"{location}: failure_type must be a string")
    for name in ("hypotheses", "evidence_refs", "signals"):
        if not isinstance(value.get(name, []), list):
            raise ValueError(f"{location}: {name} must be an array")
    for hypothesis in value["hypotheses"]:
        if (
            not isinstance(hypothesis, dict)
            or not isinstance(hypothesis.get("hypothesis_id"), str)
            or not hypothesis["hypothesis_id"].strip()
            or not isinstance(hypothesis.get("scope"), str)
            or hypothesis["scope"] not in SuggestedScope
            or not isinstance(hypothesis.get("advice"), str)
        ):
            raise ValueError(f"{location}: invalid diagnostic hypothesis")
    for ref in value["evidence_refs"]:
        if (
            not isinstance(ref, dict)
            or not isinstance(ref.get("path"), list)
            or any(type(part) not in (str, int) for part in ref["path"])
        ):
            raise ValueError(f"{location}: invalid evidence reference")
    if any(not isinstance(signal, dict) for signal in value.get("signals", [])):
        raise ValueError(f"{location}: each signal must be an object")
    return value


def load_diagnosis_reports(path: str | Path) -> list[dict[str, Any]]:
    """Load diagnoses.jsonl or one DiagnosisReport stored as JSON.

    Report order, identifiers, null values and all hypotheses are preserved.
    Filtering for prompt optimization belongs to the optimizer.
    """
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".jsonl":
        entries = [
            (number, line) for number, line in enumerate(text.splitlines(), 1)
            if line.strip()
        ]
    else:
        entries = [(1, text)]
    if not entries or not text.strip():
        raise ValueError(f"{source}: diagnostic report is empty")

    reports = []
    for number, entry in entries:
        location = f"{source}, line {number}" if source.suffix.lower() == ".jsonl" else str(source)
        try:
            value = json.loads(entry)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{location}: invalid JSON: {exc.msg}") from exc
        reports.append(_validate_report(value, location))
    return reports
