"""CLI for the agent runner: python -m app --adapter jsonl|madqa ..."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

from .runner import AgentRunner, MADQAAdapter, SubprocessJSONLAdapter, load_hf_questions, load_questions
from .runner.infrastructure.model_server import ManagedModelServer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project 180: generic agent batch runner")
    parser.add_argument("--adapter", choices=("jsonl", "madqa"), required=True)
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--questions", type=Path, help="JSON/JSONL file")
    data.add_argument("--dataset", help="Optional Hugging Face dataset, e.g. OxRML/MADQA")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-fast", action="store_true", help="Stop batch after first question error")
    # JSONL subprocess agents: any programming language, as long as it speaks protocol.
    parser.add_argument("--agent-command", help="Command for JSONL agent (no shell expansion)")
    parser.add_argument("--agent-cwd", type=Path)
    # MADQA-specific configuration lives ONLY in its adapter.
    parser.add_argument("--agent-dir", type=Path)
    parser.add_argument("--ocr-file", type=Path, default=Path("data/ocr_output.jsonl"))
    parser.add_argument("--model")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--agent-startup-timeout", type=float, default=180.0)
    parser.add_argument("--question-timeout", type=float, default=300.0)
    # An optional LLM service. If no command, we never own or stop a server.
    parser.add_argument("--model-server-command", help="Owned service command, e.g. llama-server ...")
    parser.add_argument("--model-health-url", help="Readiness endpoint; required with a model server command")
    parser.add_argument("--model-server-cwd", type=Path)
    parser.add_argument("--model-startup-timeout", type=float, default=120.0)
    parser.add_argument("--shutdown-timeout", type=float, default=5.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.limit is not None and args.limit <= 0:
            raise ValueError("--limit must be positive")
        rows = load_questions(args.questions, limit=args.limit) if args.questions else load_hf_questions(args.dataset, args.split, limit=args.limit)
        if args.adapter == "jsonl":
            if not args.agent_command:
                raise ValueError("--agent-command required for --adapter jsonl")
            command = shlex.split(args.agent_command)
            adapter_factory = lambda: SubprocessJSONLAdapter(
                command, cwd=args.agent_cwd,
                startup_timeout=args.agent_startup_timeout,
                question_timeout=args.question_timeout,
                shutdown_timeout=args.shutdown_timeout,
            )
        else:
            if not args.agent_dir or not args.model:
                raise ValueError("--adapter madqa requires --agent-dir and --model")
            adapter_factory = lambda: MADQAAdapter(
                agent_dir=args.agent_dir,
                ocr_file=args.ocr_file,
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
                max_iterations=args.max_iterations,
                top_k=args.top_k,
                startup_timeout=args.agent_startup_timeout,
                question_timeout=args.question_timeout,
                shutdown_timeout=args.shutdown_timeout,
            )
        if args.model_server_command and not args.model_health_url:
            raise ValueError("--model-health-url is required with --model-server-command")
        service = None
        if args.model_health_url:
            service = ManagedModelServer(
                command=shlex.split(args.model_server_command) if args.model_server_command else None,
                health_url=args.model_health_url,
                api_key=args.api_key,
                cwd=args.model_server_cwd,
                startup_timeout=args.model_startup_timeout,
                shutdown_timeout=args.shutdown_timeout,
            )
        report = AgentRunner(adapter_factory, model_service=service).run(
            rows, output_path=args.output, continue_on_error=not args.fail_fast
        )
        print(json.dumps({
            "processed": len(report.results),
            "errors": report.failed_count,
            "results_file": str(args.output),
        }, ensure_ascii=False))
        return 1 if report.failed_count else 0
    except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
        print(f"Runner failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
