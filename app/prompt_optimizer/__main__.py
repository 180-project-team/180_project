"""CLI for a real injected target adapter or a deterministic demo."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any

from app.llm import LLMService

from .generator import LLMPatchGenerator
from .models import PromptState
from .optimizer import propose_candidate
from .storage import write_json


def import_object(path: str) -> Any:
    if ":" not in path:
        raise ValueError("Entry point must use 'module:function' format")
    module_name, name = path.split(":", 1)
    return getattr(importlib.import_module(module_name), name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Propose one candidate prompt from a diagnosis")
    parser.add_argument("--prompt-state", type=Path, required=True, help="PromptState JSON file")
    parser.add_argument("--diagnosis", type=Path, required=True, help="DiagnosisReport JSON file")
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--model", help="Model served by an OpenAI-compatible endpoint")
    model.add_argument("--generator", help="module:function returning a patch generator")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--output", type=Path, default=Path("runs/prompt_optimizer/candidate.json"))
    args = parser.parse_args(argv)

    state = PromptState.from_dict(json.loads(args.prompt_state.read_text(encoding="utf-8")))
    diagnosis = json.loads(args.diagnosis.read_text(encoding="utf-8"))
    if not isinstance(diagnosis, dict) or not diagnosis.get("suggested_scope"):
        parser.error("--diagnosis must contain a DiagnosisReport with suggested_scope")

    proposal = None
    if diagnosis["suggested_scope"] == "PROMPT":
        if args.generator:
            proposal = propose_candidate(state, diagnosis, import_object(args.generator)())
        else:
            with LLMService(
                model=args.model,
                base_url=args.base_url,
                api_key=os.environ.get(args.api_key_env, "EMPTY"),
            ) as llm:
                proposal = propose_candidate(state, diagnosis, LLMPatchGenerator(llm))

    if proposal is None:
        write_json(args.output, {"prompt_state": None, "patch": None})
    else:
        candidate, patch = proposal
        write_json(args.output, {"prompt_state": candidate.to_dict(), "patch": patch.to_dict()})
    print(args.output)
    return 0




if __name__ == "__main__":
    raise SystemExit(main())
