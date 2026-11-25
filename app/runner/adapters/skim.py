"""Launch SkimSearchAgent in its own Python environment."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .subprocess import SubprocessJSONLAdapter


class SkimSearchAdapter(SubprocessJSONLAdapter):
    """Run one Skim worker and reuse its corpus index for the whole batch."""

    def __init__(
        self,
        corpus_path: str | Path,
        *,
        index_dir: str | Path,
        python: str | Path = sys.executable,
        strategy: str = "sieve_bm25",
        model: str | None = None,
        api_base: str = "http://127.0.0.1:8000/v1",
        max_steps: int = 50,
        startup_timeout: float = 180.0,
        question_timeout: float = 300.0,
        shutdown_timeout: float = 5.0,
    ) -> None:
        if max_steps < 1:
            raise ValueError("Skim max_steps must be positive")

        executable = os.fspath(python)
        if os.path.dirname(executable):
            executable = os.path.abspath(executable)
        command = [
            executable, "-u", "-m", "app.runner.adapters._skim_worker",
            "--corpus", str(Path(corpus_path).resolve()),
            "--index-dir", str(Path(index_dir).resolve()),
            "--strategy", strategy,
            "--api-base", api_base,
            "--max-steps", str(max_steps),
        ]
        if model is not None:
            command.extend(["--model", model])

        super().__init__(
            command,
            cwd=Path(__file__).resolve().parents[3],
            startup_timeout=startup_timeout,
            question_timeout=question_timeout,
            shutdown_timeout=shutdown_timeout,
        )
