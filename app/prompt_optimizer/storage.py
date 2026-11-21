"""Folder-based experiment history with atomic JSON artifacts."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import PromptState


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


class ExperimentStore:
    def __init__(self, root: str | Path = "runs/prompt_optimizer") -> None:
        self.root = Path(root).expanduser().resolve()

    def create_session(self) -> Path:
        identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
        directory = self.root / identifier
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "versions").mkdir()
        (directory / "iterations").mkdir()
        (directory / "agent_runs").mkdir()
        return directory

    @staticmethod
    def version(directory: Path, state: PromptState) -> None:
        path = directory / "versions" / f"{state.prompt_version_id}.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != state.to_dict():
                raise ValueError("Conflicting lineage for identical prompt content")
            return
        write_json(path, state.to_dict())
