# Prompt optimizer MVP — add-on for 180_project

Prepared against `docs/prompt_optimizer.md` and the existing `app.runner.AgentRunner` / `app.llm.LLMService` interfaces. All delivered Python files are **new**: nothing in the existing `app/runner`, `app/llm`, `app/cli.py`, evaluator, or `pyproject.toml` needs to be replaced.

## Installing on your feature branch

1. Verify you are on `feature/prompt-optimizer`: `git branch --show-current`.
2. Back up your working tree or verify `git status` is clean.
3. Extract the ZIP **into the root of `180_project`**, preserving the `app/`, `tests/`, and `examples/` folders. Do not replace existing files if the extractor asks.
4. Run from the repository root:

```bash
python -m unittest discover -s tests -p 'test_prompt_optimizer.py' -v
```

## Try it without downloading a model

```bash
python -m app.prompt_optimizer \
  --prompt-state examples/prompt_optimizer_state.json \
  --optimize-questions examples/prompt_optimizer_train.jsonl \
  --validation-questions examples/prompt_optimizer_validation.jsonl \
  --target examples.prompt_optimizer_example:make_adapter \
  --generator examples.prompt_optimizer_example:make_generator \
  --max-iterations 2 \
  --candidates-per-iteration 1
```

It intentionally starts with an ineffective prompt and produces a better prompt in an experiment folder under `runs/prompt_optimizer/`.

## Use your LLM and target agent

The CLI can call the existing shared `LLMService` via `--model <model-name> --base-url http://127.0.0.1:8000/v1` (point to an **already running** compatible model server; `LLM_API_KEY` may contain the endpoint key). Remove `--generator` when using `--model`.

The `--target your_module:make_adapter` entry point should expose:

```python
def make_adapter(prompt_state: PromptState, seed: int) -> AgentAdapter:
    # Build a fresh adapter that binds the supplied PromptState to the agent.
    # The returned adapter implements start(), ask(question), stop().
    ...
```

**Important:** An adapter must really apply the supplied prompt. Merely passing prompt text as metadata or editing a file after startup is insufficient for an optimization experiment. If your current MADQA adapter does not yet accept prompt components, implement that binding separately before running real experiments. Do not edit target agent code in this optimizer.

`RunnerEvaluationHarness(..., scorer=callable)` permits substituting your team's evaluator. The built-in fallback uses normalized exact match against `answer_variants`, `reference_answers`, or `reference_answer`. The harness uses the existing `AgentRunner` and stores its run folders intact.

## Storage

Each session saves `session.json`, `versions/*.json`, `reports/{baseline,iteration-*}/{optimize,validation}.json`, `iterations/*/diagnosis.json`, `iterations/*/candidates.json`, and original `agent_runs/<run_id>/` directories. Rejected changes stay in `candidates.json`, and generated candidate versions stay under `versions/`.

## Current limitations (deliberate MVP scope)

- Does not implement target-specific prompt binding, your team's production evaluator, or real-model inference configuration. These are injected through existing services/ports.
- Attribution is conservative and heuristic; missing or failing tools are marked `NOT_PROMPT_FIXABLE`. It does not interpret full tool-call trajectories yet.
- Supported patch mutations: `REPLACE_COMPONENT`, `APPEND_RULE`, `UPDATE_RULE`, `REMOVE_RULE`. Few-shot add/remove and complex structured prompt schemas are not implemented yet.
- No resume from an interrupted optimization session, no final test-split evaluation, no repeat-based statistical acceptance, and no latency ceiling yet.
- Production use needs a real train/validation split, a reliable scoring function, and an adapter that actually applies all advertised editable components.

## Review before committing

```bash
git status
git diff --stat
git add app/prompt_optimizer tests/test_prompt_optimizer.py examples/prompt_optimizer_example.py examples/prompt_optimizer_state.json examples/prompt_optimizer_train.jsonl examples/prompt_optimizer_validation.jsonl
git commit -m "Add prompt optimizer MVP"
git push
```

`git add` intentionally names only the new source, test, and example files; do not add outputs from `runs/` or secrets.
