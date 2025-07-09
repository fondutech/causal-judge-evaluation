# CLAUDE.md

Core guidance for Claude Code when working with the CJE repository.

Last updated: 2025-01-09 - Fixed token boundary bug causing extreme weights

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
- **Token Boundary Bug Fixed**: Teacher forcing now detects and handles tokenization boundary issues

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

## API Keys (CRITICAL)
**Always source secrets before running scripts that need API access:**
```bash
source /Users/eddielandesberg/PycharmProjects/causal-judge-evaluation/set_secrets.sh
```
This sets: FIREWORKS_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY

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

**Silent Failure Bug (Fixed Dec 2024)**: Token boundary detection could fail catastrophically
- **Symptom**: Log probs of -1500 instead of -30 (returns full sequence instead of response only)
- **Cause**: `token_counting` method assumes deterministic tokenization across API calls
- **Fix**: Use `continuation` method as primary (computes log P(full) - log P(prompt))

```python
# Implementation uses methods in order of reliability:
# 1. continuation: Most reliable, uses log subtraction
# 2. token_counting: Can fail if tokenization varies between calls
# 3. echo_based: Not implemented for most providers

from cje.utils import RobustTeacherForcing
tf = RobustTeacherForcing(provider="fireworks", model="...", temperature=0.5)
result = tf.compute_log_prob(prompt, response)  # Uses continuation method first
```

**Critical Validations**:
- Avg log prob per token should be > -10 (else likely wrong tokens)
- Response token count should match response length (~4 chars/token)
- Extreme negative values indicate boundary detection failure
- **NEVER allow silent failures** - wrong log probs corrupt all downstream analysis

**Historical Issues**:
- Token boundary misalignment causing 0.0 log probs
- Response text absorbed into prompt tokens
- Non-deterministic tokenization between API calls

**Teacher Forcing Robustness**:
- Edge case detection for problematic token boundaries
- Automatic method switching (continuation vs token counting)
- Validation and rejection of extreme importance weights

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

### API Non-Determinism Detection
**Known Issue**: Fireworks API returns different log probabilities for identical inputs
- Affects importance weights even for identical policies (pi_clone vs p0)
- Use `detect_api_nondeterminism()` to check for this issue
- Consider averaging multiple API calls for critical comparisons

```python
from cje.utils.weight_diagnostics import detect_api_nondeterminism
results = detect_api_nondeterminism(data)
if results["detected"]:
    print("API non-determinism detected:", results["reasons"])
```

## Arena 10K Experiment

**Location**: `experiments/arena_10k_oracle/`

**Key Points**:
- 4 target policies: pi_clone (baseline), pi_cot, pi_bigger_model, pi_bad
- Token boundary bug fixed with edge case detection
- Extreme weight validation in 02b_compute_logprobs.py
- English-only filter for prompts
- Pipeline resumes by default

**Running the Experiment**:
```bash
cd experiments/arena_10k_oracle/phase1_dataset_preparation
source /Users/eddielandesberg/PycharmProjects/causal-judge-evaluation/set_secrets.sh
python run_phase1_pipeline.py  # Defaults to 10k samples

# Phase 2 analysis
cd ../phase2_cje_ablations
python run_cje_analysis.py
```

**Validation Checklist**:
- pi_clone median weight should be ~1.0
- Extreme weights (>150x) automatically rejected
- Check extreme_weights.jsonl for flagged samples
- ESS should be >50% after validation

## Not Currently Supported
- Trajectory sampling (removed)
- Agent/tool-based policies  
- Multi-step reasoning traces
- PolicyRunner class (removed - use APIPolicyRunner)