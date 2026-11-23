"""Worker for the JSONL subprocess protocol.

STDOUT is reserved for the JSONL protocol, all regular prints go to STDERR.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


PROTOCOL_STDOUT = sys.stdout
sys.stdout = sys.stderr


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, default=str), file=PROTOCOL_STDOUT, flush=True)


def create_madqa_agent(args: argparse.Namespace) -> Any:
    agent_dir = Path(args.agent_dir).expanduser().resolve()
    source = agent_dir / "vllm_agent.py"
    if not source.is_file() or not (agent_dir / "utils.py").is_file():
        raise FileNotFoundError(f"MADQA files vllm_agent.py/utils.py not found in {agent_dir}")
    ocr_file = Path(args.ocr_file).expanduser()
    if not ocr_file.is_absolute():
        ocr_file = agent_dir / ocr_file
    if not ocr_file.is_file():
        raise FileNotFoundError(f"MADQA OCR data not found: {ocr_file}")
    sys.path.insert(0, str(agent_dir))
    spec = importlib.util.spec_from_file_location("_project180_madqa", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    search_module = sys.modules.get("utils")
    if search_module is None or Path(getattr(search_module, "__file__", "")).resolve() != agent_dir / "utils.py":
        raise ImportError("MADQA utils.py import conflict")
    engine = search_module.WhooshSearchEngine(str(ocr_file))
    return module.VLLMSearchAgent(engine, model=args.model, base_url=args.base_url, api_key=args.api_key)


def close_agent(agent: Any) -> None:
    if agent is None:
        return
    closer = getattr(agent, "close", None)
    if callable(closer):
        closer()
        return
    for resource in (getattr(agent, "client", None), getattr(agent, "search_engine", None)):
        closer = getattr(resource, "close", None)
        if callable(closer):
            closer()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-dir", type=Path, required=True)
    parser.add_argument("--ocr-file", type=Path, default=Path("data/ocr_output.jsonl"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    agent = None
    try:
        try:
            agent = create_madqa_agent(args)
            emit({"ready": True})
        except Exception as exc:
            emit({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
            return 2
        for line in sys.stdin:
            if not line.strip():
                continue
            request_id = None
            try:
                request = json.loads(line)
                request_id = request["id"]
                question = request["question"]
                if not isinstance(question, str) or not question.strip():
                    raise ValueError("'question' must be a non-empty string")
                result = agent.answer_question(question, max_iterations=args.max_iterations, top_k=args.top_k)
                if not isinstance(result, dict):
                    raise TypeError("MADQA answer_question() must return a dict")
                # Request id is reserved for the transport; original dataset id
                # is attached later by AgentRunner.
                emit({"id": request_id, **{k: v for k, v in result.items() if k != "id"}})
            except Exception as exc:
                emit({"id": request_id, "error": f"{type(exc).__name__}: {exc}"})
        return 0
    finally:
        close_agent(agent)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
