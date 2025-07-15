# CLAUDE.md

Core guidance for Claude Code when working with the CJE repository.

Last updated: 2025-01-10 - Arena 10K now uses deterministic llama.cpp teacher forcing

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

**Two Implementations Available**:

### 1. API-Based (RobustTeacherForcing)
```python
from cje.utils import RobustTeacherForcing
tf = RobustTeacherForcing(provider="fireworks", model="...", temperature=0.5)
result = tf.compute_log_prob(prompt, response)  # Uses continuation method
```
- Uses continuation method: log P(full) - log P(prompt)
- Subject to API non-determinism
- Good for general use, but may have variance issues

### 2. Llama.cpp (LlamaCppTeacherForcing) - RECOMMENDED for experiments
```python
from cje.utils import LlamaCppTeacherForcing
tf = LlamaCppTeacherForcing(
    model_path="models/Llama-3.2-3B-Instruct-Q6_K.gguf",
    temperature=0.5,
    seed=42
)
result = tf.compute_log_prob(prompt, response)
```
- 100% deterministic with fixed seed
- No API costs or rate limits
- GPU accelerated (Metal/CUDA)
- Used by Arena 10K experiment

**Critical Validations**:
- Avg log prob per token should be > -10 (else likely wrong tokens)
- With llama.cpp, pi_clone weights should be exactly 1.0
- Monitor extreme weights (>5x for pi_clone indicates issues)

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

### API Non-Determinism (SOLVED)
**Historical Issue**: API providers return different log probabilities for identical inputs
- **Solution**: Use llama.cpp for deterministic teacher forcing
- Arena 10K experiment now uses llama.cpp exclusively
- Pi_clone weights are exactly 1.0 with deterministic computation

## Arena 10K Experiment

**Location**: `experiments/arena_10k_oracle/`

**Key Points**:
- Uses llama.cpp exclusively for deterministic teacher forcing
- 2 target policies: pi_clone (baseline), pi_bad (unhelpful)
- Extreme weight validation: flags pi_clone weights >5x
- English-only filter for prompts
- Pipeline resumes by default
- No API costs for teacher forcing!

**Setup**:
```bash
# Install llama.cpp
pip install llama-cpp-python

# Download model (~2.5GB)
cd experiments/arena_10k_oracle
mkdir -p models
curl -L -o models/Llama-3.2-3B-Instruct-Q6_K.gguf \
  https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q6_K.gguf
```

**Running the Experiment**:
```bash
cd phase1_dataset_preparation
source /Users/eddielandesberg/PycharmProjects/causal-judge-evaluation/set_secrets.sh  # Still need for judge/oracle
python run_phase1_pipeline.py  # Defaults to 10k samples

# Phase 2 analysis
cd ../phase2_cje_ablations
python run_cje_analysis.py
```

**Validation Checklist**:
- pi_clone weights should be exactly 1.0 (deterministic!)
- Extreme weights (>5x) logged in extreme_weights.jsonl
- ESS should be near 100% with deterministic computation

## Not Currently Supported
- Trajectory sampling (removed)
- Agent/tool-based policies  
- Multi-step reasoning traces
- PolicyRunner class (removed - use APIPolicyRunner)