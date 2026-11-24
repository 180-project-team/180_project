# Prompt optimizer 

The optimizer proposes a prompt change from a Diagnostics report and compares prompt versions using scores from the shared Evaluator.

## Run the tests and demo

Run these commands from the repository root.

```bash
python3 -m unittest discover -s tests
python3 -m examples.prompt_optimizer_example
```

The demo uses arithmetic questions, a sample diagnosis, and simulated scores. It saves the proposed prompt, the patch, and the selection decision to `runs/prompt_optimizer_demo/candidate.json`.

## Propose a change with your LLM

Pass the current prompt state and a Diagnostics report as JSON files. Set `--base-url` to your running model server.

```bash
python3 -m app.prompt_optimizer \
  --prompt-state path/to/prompt_state.json \
  --diagnosis path/to/diagnosis.json \
  --model qwen3:8b \
  --base-url http://127.0.0.1:11434/v1 \
  --output runs/prompt_optimizer/candidate.json
```

Set `LLM_API_KEY` if your server requires authentication. The command saves `prompt_state` and `patch`. Both fields are `null` when the report has no applicable prompt recommendation.

## Call from the orchestrator

- `propose_candidate(state, diagnosis, generator)` returns a candidate prompt and patch for an applicable recommendation.
- `load_evaluation_report(...)` joins saved Runner results and Evaluator scores by question ID.
- `SelectionPolicy.compare(baseline, candidate)` uses Fact Recall to compare prompt versions on the same questions, with configurable limits on regressions and token usage.

The orchestrator supplies the diagnosis and runs each prompt version through Runner and Evaluator. The agent adapter applies the corresponding prompt before each run.
