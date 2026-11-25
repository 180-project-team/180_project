"""Standalone runner CLI. Production orchestration can call AgentRunner directly."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from . import AgentRunner, SkimSearchAdapter, SubprocessJSONLAdapter, load_hf_questions, load_questions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project 180: execute a target agent on a batch of questions")
    parser.add_argument("--adapter", choices=("jsonl", "skim"), default="jsonl")
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--questions", type=Path, help="JSON/JSONL file")
    data.add_argument("--dataset", help="Optional Hugging Face dataset ID")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--run-dir", type=Path, default=Path("."), help="Caller-supplied run directory")
    parser.add_argument("--results-path", type=Path, help="Write runner results to this JSONL file")
    parser.add_argument("--fail-fast", action="store_true")
    agent = parser.add_mutually_exclusive_group(required=True)
    agent.add_argument("--agent-command", help="Command for JSONL subprocess agent, without shell expansion")
    agent.add_argument("--skim-corpus", type=Path, help="SkimSearchAgent document corpus in JSONL format")
    parser.add_argument("--agent-cwd", type=Path)
    parser.add_argument("--skim-python", default=sys.executable, help="Python executable with SkimSearchAgent installed")
    parser.add_argument("--skim-strategy", default="sieve_bm25")
    parser.add_argument("--skim-model", help="Model ID for SkimSearchAgent (optional for smoke tests)")
    parser.add_argument("--skim-api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--skim-max-steps", type=int, default=50)
    parser.add_argument("--agent-startup-timeout", type=float, default=180.0)
    parser.add_argument("--question-timeout", type=float, default=300.0)
    parser.add_argument("--shutdown-timeout", type=float, default=5.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.adapter == "skim" and args.skim_corpus is None:
        parser.error("--adapter skim requires --skim-corpus")
    if args.adapter == "jsonl" and args.agent_command is None:
        parser.error("--adapter jsonl requires --agent-command")

    try:
        if args.limit is not None and args.limit <= 0:
            raise ValueError("--limit must be positive")
        if args.results_path is not None and any(
            args.results_path.resolve() == path.resolve()
            for path in (args.questions, args.skim_corpus) if path is not None
        ):
            raise ValueError("--results-path must differ from the input files")
        rows = load_questions(args.questions, limit=args.limit) if args.questions else load_hf_questions(args.dataset, args.split, limit=args.limit)

        def adapter_factory(run_dir: Path) -> SubprocessJSONLAdapter:
            timeouts = {
                "startup_timeout": args.agent_startup_timeout,
                "question_timeout": args.question_timeout,
                "shutdown_timeout": args.shutdown_timeout,
            }
            if args.adapter == "skim":
                return SkimSearchAdapter(
                    corpus_path=args.skim_corpus,
                    index_dir=run_dir / "skim_indexes",
                    python=args.skim_python,
                    strategy=args.skim_strategy,
                    model=args.skim_model,
                    api_base=args.skim_api_base,
                    max_steps=args.skim_max_steps,
                    **timeouts,
                )
            return SubprocessJSONLAdapter(
                shlex.split(args.agent_command),
                cwd=args.agent_cwd,
                env={"AGENT_RUN_DIR": str(run_dir.resolve())},
                **timeouts,
            )

        runner = AgentRunner(adapter_factory)
        failed = False
        for row in runner.run(
            rows, args.run_dir, results_path=args.results_path, continue_on_error=not args.fail_fast,
        ):
            print(json.dumps(row, ensure_ascii=False, default=str), flush=True)
            failed |= row["status"] == "error"
        return int(failed)
    except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
        print(f"Runner failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
