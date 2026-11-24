"""Standalone runner CLI. Production orchestration can call AgentRunner directly."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from . import AgentRunner, SubprocessJSONLAdapter, load_hf_questions, load_questions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project 180: execute a target agent on a batch of questions")
    parser.add_argument("--adapter", choices=("jsonl",), default="jsonl")
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--questions", type=Path, help="JSON/JSONL file")
    data.add_argument("--dataset", help="Optional Hugging Face dataset ID")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--run-dir", type=Path, default=Path("."), help="Caller-supplied run directory")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--agent-command", required=True, help="Command for JSONL subprocess agent, without shell expansion")
    parser.add_argument("--agent-cwd", type=Path)
    parser.add_argument("--agent-startup-timeout", type=float, default=180.0)
    parser.add_argument("--question-timeout", type=float, default=300.0)
    parser.add_argument("--shutdown-timeout", type=float, default=5.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.limit is not None and args.limit <= 0:
            raise ValueError("--limit must be positive")
        rows = load_questions(args.questions, limit=args.limit) if args.questions else load_hf_questions(args.dataset, args.split, limit=args.limit)
        command = shlex.split(args.agent_command)

        def adapter_factory(run_dir: Path) -> SubprocessJSONLAdapter:
            return SubprocessJSONLAdapter(
                command,
                cwd=args.agent_cwd,
                env={"AGENT_RUN_DIR": str(run_dir.resolve())},
                startup_timeout=args.agent_startup_timeout,
                question_timeout=args.question_timeout,
                shutdown_timeout=args.shutdown_timeout,
            )

        runner = AgentRunner(adapter_factory)
        failed = False
        for row in runner.run(rows, args.run_dir, continue_on_error=not args.fail_fast):
            print(json.dumps(row, ensure_ascii=False, default=str), flush=True)
            failed |= row["status"] == "error"
        return int(failed)
    except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
        print(f"Runner failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
