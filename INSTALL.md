# Prompt optimizer

The L1 optimizer proposes a prompt-only patch from an existing Diagnostics report
and an agent source snapshot. The caller runs the candidate through the shared
Runner and Evaluator and decides whether to accept it.

## Environment and checks

From the repository root:

```shell
uv sync --locked --python 3.12
uv run --locked --python 3.12 python -m unittest discover -s tests -v
uv run --locked --python 3.12 python -m examples.prompt_optimizer_interface
```

The offline demo prepares a report using the team's Diagnostics and uses a fixed
patch generator. It saves `patch.json`, `candidate_code.txt`, and `changes.diff`
under `runs/prompt_optimizer_interface`. It checks artifact exchange without
measuring a quality improvement. The previous `examples.prompt_optimizer_example`
entry point runs this same demo.

## Use your running model

```shell
uv run --locked --python 3.12 python -m app.prompt_optimizer \
  --diagnosis path/to/diagnoses.jsonl \
  --output-dir runs/prompt_candidate \
  --agent-code-file path/to/agent_snapshot.txt \
  --model qwen3:8b --base-url http://127.0.0.1:11434/v1
```

The snapshot file contains the plain string with the agent structure and code.
Alternatively pass that string via `--agent-code`. The CLI reads `LLM_API_KEY`
unless another environment variable is selected with `--api-key-env`.

## Use the orchestrator's shared client

```python
from app.prompt_optimizer import PromptOptimizer

result = PromptOptimizer(llm_service).optimize(diagnosis_path, output_dir, agent_code)
```

The caller owns the shared client's lifecycle. `proposed` means a locally checked
candidate requiring external evaluation. The optimizer does not execute the agent,
calculate metrics, run diagnostics, or apply changes to its source files.

The exact patch format, supported Python prompt literals, secondary hypotheses,
and current integration limits are documented in
[docs/prompt_optimizer.md](docs/prompt_optimizer.md) and
[docs/prompt_optimizer_interface.md](docs/prompt_optimizer_interface.md).
