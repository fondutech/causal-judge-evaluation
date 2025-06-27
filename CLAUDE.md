# CLAUDE.md

Core guidance for Claude Code when working with the CJE repository.

## 🎯 Hygiene Rules

### STARTUP
1. Check git status and recent commits
2. Run `python scripts/hygiene_check.py`
3. Run `make lint`

### AUTO-FIX
Fix immediately without asking:
- Commands/features that don't exist
- References to deleted files/modules
- Documentation contradicting implementation
- Outdated comments or docstrings

### CRITICAL: NO FALLBACK VALUES
- NEVER use fallback values for log probabilities (no -100.0, 0.0, etc.)
- All failures must return None/null and be handled explicitly
- Use LogProbResult type for all log probability computations

## Essential Commands
```bash
make dev-setup                 # Initial setup
poetry run pytest              # Run tests (fast)
poetry run pytest --run-slow   # Include slow tests
cje run --cfg-path configs --cfg-name example_eval  # Run experiment via CLI
make lint                      # MUST pass before ANY commit

# Python API (new modular pipeline):
from cje.config.unified import simple_config
config = simple_config(dataset_name="test.jsonl", ...)
results = config.run()
```

## Architecture

**Pipeline**: Data → Log Probs → Judge → Calibrate → Estimate → Results

**Key Principles**:
- All judges return `JudgeScore(mean, variance)`
- All policies return `LogProbResult` (never raw floats)
- Single source of truth - no duplicates
- Uncertainty built-in from the start
- Explicit error handling - no silent failures

**Implementation**:
- Teacher forcing for unbiased log probabilities
- Cross-fitting k≥2 (prevents overfitting)
- Log ratio clipping at ±20.0
- Scores stored as `{"mean": x, "variance": y}`
- Failed log probs stored as null with error details

## Judge System
- Three uncertainty methods: deterministic, confidence_interval, monte_carlo
- Provider abstraction with capability tracking
- Fireworks/Together: full teacher forcing
- OpenAI/Anthropic: judge-only

## Oracle Labeling
Use `cje.oracle_labeling` module for ground truth:
```python
from cje.oracle_labeling import add_oracle_labels
rows_with_oracle = add_oracle_labels(rows, provider="openai", model_name="gpt-4o")
```

For judge scoring, use CJE's judge system directly (see experiments/arena_10k_oracle/phase1_dataset_preparation/04*.py)

## Type System & Error Handling

**Core Types**:
- `LogProbResult`: Wraps log probability computations with status/error info
- `Result<T>`: Generic result type for safe error handling
- `SampleResult/BatchResult`: Structured results for multi-target sampling

**Policy System**:
- All policies inherit from `BasePolicy` 
- Must implement `_compute_log_prob_impl()` (can raise exceptions)
- Base class handles retries, error tracking, and returns `LogProbResult`
- Use `create_api_policy()` factory for API-based policies

**Error Handling**:
- NO fallback values (no -100.0, 0.0, etc.)
- Failed computations return explicit None/null
- Rich error context with retry counts and error types
- Use `@with_retry` decorator for automatic retry logic

**Backfilling Log Probs**:
```bash
cje backfill-logp input.jsonl output.jsonl --provider openai --model gpt-3.5-turbo
```
Failed items have `logp: null` and `logp_error` field with details.

## Not Currently Supported
- Trajectory sampling (removed `trajectory_sampler.py`)
- Agent/tool-based policies
- Multi-step reasoning traces
