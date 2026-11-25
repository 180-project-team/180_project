"""Serve a SkimSearchAgent instance over the Runner's JSONL protocol."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, TextIO


def _load_corpus(path: Path) -> list[dict[str, Any]]:
    documents = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                document = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(document, dict):
                raise ValueError(f"Expected a document object on line {line_number} of {path}")
            if document.get("docid") is not None:
                document.setdefault("doc_id", str(document["docid"]))
            documents.append(document)
    return documents


def _serve(args: argparse.Namespace, protocol: TextIO) -> int:
    def send(message: dict[str, Any]) -> None:
        print(json.dumps(message, ensure_ascii=False), file=protocol, flush=True)

    try:
        from agent_search import as_units, build_agent

        units = as_units(_load_corpus(args.corpus))
        if not units:
            raise ValueError(f"No indexable documents in {args.corpus}")
        agent = build_agent(
            args.strategy,
            model=args.model,
            backend="api",
            api_base=args.api_base,
            max_steps=args.max_steps,
            index_root=args.index_dir,
        )
        agent.index(units)
    except Exception as exc:
        send({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1

    send({"ready": True})
    for line in sys.stdin.buffer:
        request = json.loads(line)
        response = {"id": request["id"]}
        try:
            ranking = agent.search(request["question"], k=10)
            trace = agent.last_trajectory_meta
            response.update(
                answer=trace["final_answer"],
                trajectory=trace["trajectory"],
                stopped=trace["stopped"],
                ranking=ranking,
            )
        except Exception as exc:
            response["error"] = f"{type(exc).__name__}: {exc}"
        send(response)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--model")
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    args = parser.parse_args()

    # Keep library and JVM output out of the JSONL pipe.
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8") as protocol:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        return _serve(args, protocol)


if __name__ == "__main__":
    raise SystemExit(main())
