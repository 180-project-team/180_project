"""Minimal independent local agent for smoke tests (no GPU, no dependencies)."""

import json
import sys

print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    if not line.strip():
        continue
    request = json.loads(line)
    print(json.dumps({
        "id": request["id"],
        "answer": request["question"].upper(),
        "citations": [],
        "trajectory": [],
    }, ensure_ascii=False), flush=True)
