"""File-based L1 optimizer; execution, evaluation and acceptance stay outside."""

from __future__ import annotations

import copy
import difflib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from app.prompt_optimizer.diagnostic_report import load_diagnosis_reports, prompt_hypotheses
from app.prompt_optimizer.models import digest
from app.prompt_optimizer.source_generator import LLMSourcePatchGenerator, SourcePatchGenerator
from app.prompt_optimizer.source_patch import SourcePromptPatch


@dataclass(frozen=True)
class OptimizationResult:
    status: str
    reason: str
    result_path: Path
    patch: SourcePromptPatch | None
    candidate_code: str


class PromptOptimizer:
    def __init__(
        self, llm_service: Any = None, *, generator: SourcePatchGenerator | None = None,
        primary_scope_only: bool = False,
    ):
        if (llm_service is None) == (generator is None):
            raise ValueError("Provide exactly one llm_service or generator")
        self.generator = generator if generator is not None else LLMSourcePatchGenerator(llm_service)
        self.primary_scope_only = primary_scope_only

    def optimize(
        self, diagnosis_path: str | Path, output_dir: str | Path, agent_code: str,
    ) -> OptimizationResult:
        """Propose one patch from JSON/JSONL and an opaque source snapshot string.

        The caller owns the injected LLM lifecycle. This method does not execute
        the agent, apply a patch to its files, or decide whether to accept it.
        """
        if not isinstance(agent_code, str) or not agent_code.strip():
            raise ValueError("agent_code must be a non-empty string")
        source = Path(diagnosis_path)
        reports = load_diagnosis_reports(source)
        directory = Path(output_dir)
        names = ("candidate_code.txt", "changes.diff", "patch.json")
        for name in names:
            output = directory / name
            if output.resolve() == source.resolve() or (output.exists() and output.samefile(source)):
                raise ValueError("Optimizer output overlaps the diagnostic input")
            if output.is_dir():
                raise IsADirectoryError(f"Optimizer output is a directory: {output}")

        eligible = []
        for index, report in enumerate(reports):
            hypotheses = prompt_hypotheses(report, primary_scope_only=self.primary_scope_only)
            if hypotheses:
                eligible.append({**report, "report_index": index, "hypotheses": list(hypotheses)})

        status, reason, patch = "skipped", "No usable PROMPT hypotheses with evidence", None
        selected_report = None
        if eligible:
            try:
                proposal = self.generator.generate(agent_code, copy.deepcopy(eligible))
                if not isinstance(proposal, Mapping):
                    raise ValueError("Generator must return a JSON patch object")
                if proposal.get("no_change") is True:
                    if not isinstance(proposal.get("reason"), str) or not proposal["reason"].strip():
                        raise ValueError("A no_change result requires a reason")
                    reason = proposal["reason"]
                else:
                    report_index = proposal.get("report_index")
                    report = next((row for row in eligible if type(report_index) is int and row["report_index"] == report_index), None)
                    if report is None or proposal.get("hypothesis_id") not in {item["hypothesis_id"] for item in report["hypotheses"]}:
                        raise ValueError("Patch must reference a supplied PROMPT hypothesis")
                    patch = SourcePromptPatch.create(
                        agent_code, before=proposal.get("before"), after=proposal.get("after"),
                        rationale=proposal.get("rationale"), report_index=report_index,
                        hypothesis_id=proposal["hypothesis_id"], evidence_refs=report["evidence_refs"],
                    )
                    selected_report = {name: report.get(name) for name in ("case_id", "run_id", "version_id", "report_index")}
                    status, reason = "proposed", "Prompt candidate requires external evaluation"
            except ValueError as exc:
                status, reason = "rejected", str(exc)
            except Exception as exc:
                status, reason = "error", f"Patch generation failed: {type(exc).__name__}: {exc}"

        candidate = patch.apply(agent_code) if patch is not None else agent_code
        changes = "".join(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n" for line in difflib.unified_diff(
            agent_code.splitlines(keepends=True), candidate.splitlines(keepends=True),
            fromfile="a/agent_snapshot.txt", tofile="b/agent_snapshot.txt",
        ))
        manifest = {
            "status": status, "reason": reason, "source_hash": digest(agent_code),
            "candidate_hash": digest(candidate), "diagnosis_path": str(source.resolve()),
            "reports_count": len(reports), "eligible_reports_count": len(eligible),
            "primary_scope_only": self.primary_scope_only,
            "diagnosis_ref": selected_report,
            "patch": patch.to_dict() if patch is not None else None,
            "candidate_path": "candidate_code.txt", "diff_path": "changes.diff",
        }
        directory.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=directory, prefix=".prompt-optimizer-") as temporary:
            staging = Path(temporary)
            (staging / names[0]).write_text(candidate, encoding="utf-8", newline="")
            (staging / names[1]).write_text(changes, encoding="utf-8", newline="")
            (staging / names[2]).write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            # Publish the manifest last. Use a separate output directory per call.
            for name in names:
                (staging / name).replace(directory / name)
        return OptimizationResult(status, reason, directory / "patch.json", patch, candidate)
