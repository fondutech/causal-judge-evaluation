# CLAUDE.md

Core guidance for Claude Code when working with the CJE repository.

Last updated: 2024-06-29 - Added Arena 10K experiment details

## 🎯 Hygiene Rules

### STARTUP
1. Check git status and recent commits
2. Run `make lint` before any commits
3. Run tests: `poetry run pytest`

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
- Watch for exact 0.0 log probs on non-empty responses (indicates bug)

## Essential Commands
```bash
make dev-setup                 # Initial setup
poetry run pytest              # Run tests (fast)
poetry run pytest --run-slow   # Include slow tests
cje run --cfg-path configs --cfg-name example_eval  # Run experiment via CLI
make lint                      # MUST pass before ANY commit

# Python API:
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

## Teacher Forcing (CRITICAL)

**The Bug We Fixed**: Token boundary misalignment caused 0.0 log probs
```python
# WRONG - assumes character offset = token boundary
response_tokens = [t for t in tokens if t.offset >= len(prompt)]

# RIGHT - use robust multi-method approach
from cje.utils import RobustTeacherForcing, compute_teacher_forced_logprob
```

**Key Insights**:
- Tokens don't align with text boundaries ("Say A" → ['Say', ' the', ' letter', ' A'])
- Response text can be absorbed into prompt tokens
- Models with same tokenizer fail on same inputs
- Always validate log probs (0.0 for non-empty response is suspicious)

## Judge System
- Three uncertainty methods: deterministic, confidence_interval, monte_carlo
- Provider abstraction with capability tracking
- Fireworks/Together: full teacher forcing support
- OpenAI/Anthropic: judge-only (no teacher forcing)

## Working with Precomputed Data
```python
from cje.loggers import PrecomputedSampler
from cje.estimators import CalibratedIPS

# Create sampler from JSONL with teacher-forced log probs
sampler = PrecomputedSampler.from_jsonl("data_with_logps.jsonl")

# Use with any estimator
estimator = CalibratedIPS(sampler)
estimator.fit(logs)
results = estimator.estimate()
```

## Type System & Error Handling

**Core Types**:
- `LogProbResult`: Wraps log probability computations with status/error info
- `JudgeScore`: Always includes mean AND variance
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
- Monitor importance weights (>100 or <0.01 indicates problems)

## Importance Weight Monitoring

Watch for these red flags:
```python
# Extreme weights indicate problems
if weight > 100 or weight < 0.01:
    log.warning(f"Extreme weight: {weight}")

# Check effective sample size
ess = (sum(weights))**2 / sum(w**2 for w in weights)
if ess < n_samples * 0.1:  # Less than 10% effective samples
    log.error("Effective sample size too low!")
```

## Arena 10K Experiment

**Location**: `experiments/arena_10k_oracle/`

**Key Points**:
- 4 target policies: pi_clone (baseline), pi_cot, pi_bigger_model, pi_bad
- Teacher forcing bug fixed - no more 0.0 log probs for non-empty responses
- 1% sample test before full run: `./run_sample_test.sh`
- Full run: ~140k API calls, ~$60, 50-75 hours

**Critical Validation**:
```bash
# After sample run, check teacher forcing:
python analyze_teacher_forcing_stats.py
# MUST show: "No suspicious zero values found!"
```

## Not Currently Supported
- Trajectory sampling (removed)
- Agent/tool-based policies  
- Multi-step reasoning traces
- PolicyRunner class (removed - use APIPolicyRunner)