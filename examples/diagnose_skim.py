"""Read scored Skim Search episodes and write one diagnosis per JSONL line."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from app.diagnostics import diagnose_case, skim_case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows", type=Path, help="Path to Skim Search rows.jsonl")
    args = parser.parse_args()
    with args.rows.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Expected a JSON object")
                if row.get("skipped"):
                    continue
                case = skim_case(row)
                print(json.dumps(asdict(diagnose_case(case)), ensure_ascii=False))
            except ValueError as exc:
                parser.error(f"{args.rows}, line {number}: {exc}")


if __name__ == "__main__":
    main()
