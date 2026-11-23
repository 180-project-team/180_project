from __future__ import annotations

import sys
from pathlib import Path

from .subprocess import SubprocessJSONLAdapter


class MADQAAdapter(SubprocessJSONLAdapter):
    def __init__(
        self,
        *,
        agent_dir: str | Path,
        run_dir: str | Path | None = None,
        ocr_file: str | Path = "data/ocr_output.jsonl",
        model: str,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str = "EMPTY",
        max_iterations: int = 5,
        top_k: int = 3,
        startup_timeout: float = 180.0,
        question_timeout: float = 300.0,
        shutdown_timeout: float = 5.0,
    ) -> None:
        if max_iterations <= 0 or top_k <= 0:
            raise ValueError("max_iterations and top_k must be positive")
        worker = Path(__file__).resolve().parents[1] / "workers" / "madqa_worker.py"
        command = [
            sys.executable, "-u", str(worker),
            "--agent-dir", str(Path(agent_dir).expanduser().resolve()),
            "--ocr-file", str(ocr_file),
            "--model", model,
            "--base-url", base_url,
            "--api-key", api_key,
            "--max-iterations", str(max_iterations),
            "--top-k", str(top_k),
        ]
        super().__init__(
            command,
            env={"AGENT_RUN_DIR": str(Path(run_dir).resolve())} if run_dir is not None else None,
            cwd=Path(agent_dir).expanduser().resolve(),
            startup_timeout=startup_timeout,
            question_timeout=question_timeout,
            shutdown_timeout=shutdown_timeout,
        )
