"""CLI for proposing a prompt patch from Diagnostics and agent source."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
from typing import Any

from app.llm import LLMService

from app.prompt_optimizer.service import PromptOptimizer


def import_object(path: str) -> Any:
    if ":" not in path:
        raise ValueError("Entry point must use 'module:function' format")
    module_name, name = path.split(":", 1)
    return getattr(importlib.import_module(module_name), name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Propose one prompt-only patch from existing diagnostics")
    parser.add_argument("--diagnosis", type=Path, required=True, help="diagnoses.jsonl or one DiagnosisReport JSON")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for optimizer artifacts")
    code = parser.add_mutually_exclusive_group(required=True)
    code.add_argument("--agent-code", help="Agent code and structure as a plain string")
    code.add_argument("--agent-code-file", type=Path, help="Read that string from a UTF-8 text file")
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--model", help="Model served by an OpenAI-compatible endpoint")
    model.add_argument("--generator", help="module:function returning a SourcePatchGenerator")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--primary-scope-only", action="store_true", help="Use only reports whose primary scope is PROMPT")
    args = parser.parse_args(argv)

    try:
        agent_code = args.agent_code_file.read_text(encoding="utf-8") if args.agent_code_file else args.agent_code
        if args.generator:
            optimizer = PromptOptimizer(generator=import_object(args.generator)(), primary_scope_only=args.primary_scope_only)
            result = optimizer.optimize(args.diagnosis, args.output_dir, agent_code)
        else:
            with LLMService(
                model=args.model,
                base_url=args.base_url,
                api_key=os.environ.get(args.api_key_env, "EMPTY"),
            ) as llm:
                result = PromptOptimizer(llm, primary_scope_only=args.primary_scope_only).optimize(
                    args.diagnosis, args.output_dir, agent_code,
                )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"{result.status}: {result.reason}")
    print(result.result_path)
    return 1 if result.status in ("rejected", "error") else 0

if __name__ == "__main__":
    raise SystemExit(main())
