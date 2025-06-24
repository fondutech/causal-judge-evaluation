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
- Single source of truth - no duplicates
- Uncertainty built-in from the start

**Implementation**:
- Teacher forcing for unbiased log probabilities
- Cross-fitting k≥2 (prevents overfitting)
- Log ratio clipping at ±20.0
- Scores stored as `{"mean": x, "variance": y}`

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
