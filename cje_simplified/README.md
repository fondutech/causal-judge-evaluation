# Simplified CJE (Causal Judge Evaluation)

A minimal implementation of the CJE methodology for unbiased evaluation of LLM improvements using AI judges.

## Overview

This simplified library focuses on the core CJE workflow:
1. **Load** precomputed log probabilities and judge scores
2. **Calibrate** judge scores to oracle KPIs (optional but recommended)
3. **Calibrate** importance weights using isotonic regression
4. **Estimate** unbiased policy performance with proper confidence intervals

## Installation

```bash
pip install numpy scipy scikit-learn openai
```

## Quick Start

```python
from cje_simplified import PrecomputedSampler, CalibratedIPS, RawIPS

# Load precomputed data
sampler = PrecomputedSampler.from_jsonl("data.jsonl")

# Check data quality
print(f"Total samples: {sampler.n_samples}")
print(f"Valid samples (with all log probs): {sampler.n_valid_samples}")

# Option 1: Calibrated IPS (recommended - handles extreme weights via calibration)
estimator = CalibratedIPS(sampler, clip_weight=1e10)  # No pre-clipping by default
results = estimator.fit_and_estimate()

# Option 2: Raw IPS (standard importance sampling with clipping)
estimator = RawIPS(sampler, clip_weight=100.0)  # Clips weights at 100
results = estimator.fit_and_estimate()

# Analyze results
print(f"Best policy: {sampler.target_policies[results.best_policy()]}")
print(f"95% CI: {results.confidence_interval(0.95)}")

# Access diagnostics (ESS, tail ratios, variance metrics)
diagnostics = results.metadata.get('diagnostics', {})
for policy in sampler.target_policies:
    diag = diagnostics.get(policy, {})
    print(f"{policy}: ESS={diag.get('weights', {}).get('ess_fraction', 0):.1%}")
```

## Data Preparation Workflow

### Step 1: Calibrate Judge Scores to Business KPIs

```python
from cje_simplified import load_dataset_from_jsonl, calibrate_dataset

# Load dataset with judge scores and oracle labels
dataset = load_dataset_from_jsonl("data_with_judge_scores.jsonl")

# Calibrate judge scores to oracle labels
calibrated_dataset, stats = calibrate_dataset(
    dataset,
    judge_field="gpt4_score",     # Field with judge scores
    oracle_field="human_rating"   # Field with true KPIs
)

print(f"Calibration RMSE: {stats.calibration_rmse:.3f}")
print(f"Coverage: {stats.coverage_at_01:.1%}")
```

### Step 2: Expected Data Format

After preparation, data should have this format:

```json
{
  "prompt": "What is machine learning?",
  "response": "Machine learning is...",
  "reward": 0.85,  # Calibrated reward (not raw judge score!)
  "base_policy_logprob": -35.704,
  "target_policy_logprobs": {
    "pi_cot": -40.123,
    "pi_bigger": -32.456
  }
}
```

Key requirements:
- `reward`: Calibrated reward aligned with business KPI
- `base_policy_logprob`: Log probability under base/behavior policy
- `target_policy_logprobs`: Log probabilities under each target policy
- Failed computations stored as `null` (no fallback values)

**Important**: Samples with missing log probabilities (`null` values) are automatically filtered during analysis. The system will warn you if many samples are filtered:
- Warning when >10% of samples are filtered
- Error logged when >50% of samples are filtered
- Use `sampler.n_valid_samples` to check how many samples will be used

## Computing Log Probabilities

For Fireworks API:

```python
from cje_simplified import compute_teacher_forced_logprob

result = compute_teacher_forced_logprob(
    prompt="What is 2+2?",
    response="The answer is 4.",
    model="accounts/fireworks/models/llama-v3p2-3b-instruct",
    temperature=1.0
)

if result.is_valid:
    print(f"Log prob: {result.value}")
```

Set `FIREWORKS_API_KEY` environment variable or pass `api_key` parameter.

### Chat Format Support

For chat models, use the conversion utilities with explicit template configuration:

```python
from cje_simplified import (
    convert_chat_for_teacher_forcing, 
    Llama3TemplateConfig,
    Llama4TemplateConfig,
    HuggingFaceTemplateConfig
)

# Convert chat to completions format
chat = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."}
]

# Use Llama 3 template
config = Llama3TemplateConfig()
prompt_only, prompt_plus_reply = convert_chat_for_teacher_forcing(
    chat,
    template_config=config,
    tokenizer_name="meta-llama/Llama-3.2-3B-Instruct"
)

# Or use default Llama 4 template
prompt_only, prompt_plus_reply = convert_chat_for_teacher_forcing(
    chat,
    tokenizer_name="meta-llama/Llama-4-Scout-17B-Instruct"
)

# Then compute log probs using two calls
from cje_simplified.teacher_forcing import compute_total_logprob

lp_full = compute_total_logprob(prompt_plus_reply, model="...")
lp_prefix = compute_total_logprob(prompt_only, model="...")
lp_reply = lp_full.value - lp_prefix.value
```

This handles:
- Exact chat template formatting (Llama 3, Llama 4, etc.)
- Token boundary alignment for subtraction
- Whitespace and control token edge cases

For custom templates, implement the ChatTemplateConfig ABC:

```python
from cje_simplified.teacher_forcing import ChatTemplateConfig

class CustomTemplateConfig(ChatTemplateConfig):
    """Custom chat template implementation."""
    
    def format_message(self, role: str, content: str) -> str:
        return f"<|{role}|>\n{content}<|end|>"
    
    def format_message_header(self, role: str) -> str:
        return f"<|{role}|>\n"
    
    def should_add_bos(self) -> bool:
        return False

# Use your custom template
config = CustomTemplateConfig()
prompt_only, prompt_plus_reply = convert_chat_for_teacher_forcing(
    chat, template_config=config
)
```

## Weight Diagnostics and Variance Control

The library now includes comprehensive diagnostics to assess estimation reliability:

### Diagnostics
- **Effective Sample Size (ESS)**: Measures overlap quality between policies
- **Tail Weight Ratios**: Detects extreme weights that can destabilize estimates
- **Mass Concentration**: Shows if a few samples dominate the estimate
- **Variance Metrics**: Tracks variance reduction from calibration

### Variance-Safe Calibration
CalibratedIPS now defaults to variance-safe calibration, preventing pathological variance explosions:

```python
# Default: variance-safe calibration
estimator = CalibratedIPS(sampler)  # enforce_variance_nonincrease=True by default

# Optional: pure isotonic calibration (may increase variance)
estimator = CalibratedIPS(sampler, enforce_variance_nonincrease=False)
```

### Accessing Diagnostics
```python
# From results
diagnostics = results.metadata['diagnostics']

# Or directly from estimator
diag = estimator.get_diagnostics('policy_name')
print(f"ESS: {diag['weights']['ess_fraction']:.1%}")
print(f"Status: {diag['status']}")  # green/amber/red
```

## Judge Score Calibration

Calibrate cheap judge scores to actual KPIs using oracle labels:

```python
from cje_simplified import calibrate_judge_scores

# Calibrate using 25% oracle subset
calibrated_scores, stats = calibrate_judge_scores(
    judge_scores=all_judge_scores,  # All raw scores
    oracle_labels=oracle_kpis[:1000]  # First 1000 have labels
)

print(f"Calibration RMSE: {stats['rmse']:.3f}")
print(f"Coverage: {stats['coverage']:.1%}")
```

Key features:
- Uses isotonic regression to preserve score ordering
- Cross-fitting prevents overfitting to oracle labels
- Handles partial labeling (e.g., 25% oracle subset)

## Running Ablation Studies

### Oracle Coverage Ablations

Compare estimator performance with different amounts of oracle labels:

```bash
cd cje_simplified/experiments/arena_10k_simplified

# Run ablations with default settings (25%, 50%, 100% oracle coverage)
poetry run python analyze_oracle_coverage.py --prepare-data

# Run without weight clipping to see true variance
poetry run python analyze_oracle_coverage.py --no-clipping

# Use custom clip weight
poetry run python analyze_oracle_coverage.py --clip-weight 50
```

This will:
- Test multiple oracle coverage levels (25%, 50%, 100%)
- Compare CalibratedIPS vs RawIPS estimators
- Test with and without weight clipping
- Generate a summary table showing estimates, confidence intervals, and ESS

### Arena Experiment Pipeline

Run full experiments on ChatBot Arena data:

```bash
cd cje_simplified/experiments/arena_10k_simplified

# Prepare prompts from ChatBot Arena (filters inappropriate content)
poetry run python pipeline_steps/prepare_arena_data.py --n-samples 1000

# Generate responses and compute log probabilities
poetry run python pipeline_steps/generate_responses.py --model llama-3.2-3b

# Create CJE dataset and analyze
poetry run python generate_arena_data.py --n-samples 1000
poetry run python analyze_dataset.py --data data/cje_dataset.jsonl
```

## Weight Diagnostics

Debug importance weights to catch common issues:

```python
from cje_simplified import diagnose_weights

# Get weight diagnostics
weights = estimator.get_weights("pi_cot")
diag = diagnose_weights(weights, "pi_cot")
print(diag.summary())
```

## Library Structure

```
cje_simplified/
├── core/                    # Core estimation algorithms
│   ├── calibrated_ips.py        # Main CJE estimator
│   └── types.py                 # Result types and error handling
├── data/                    # Data loading and preparation
│   ├── precomputed_sampler.py   # Load prepared data
│   └── reward_utils.py          # Calibrate judge scores
├── teacher_forcing/         # Log probability computation
│   ├── api/                     # API provider implementations
│   │   └── fireworks.py         # Fireworks teacher forcing
│   ├── templates/               # Chat template configurations  
│   │   ├── base.py              # ChatTemplateConfig ABC
│   │   ├── llama.py             # Llama template implementations
│   │   └── huggingface.py       # HuggingFace auto-detection
│   └── chat.py                  # Chat conversion utilities
└── utils/                   # Utilities and diagnostics
    ├── calibration_utils.py     # Shared isotonic regression
    ├── judge_calibration.py     # Judge → KPI mapping
    └── weight_diagnostics.py    # Debug importance weights
```

## Example

See `example_usage.py` for a complete working example.

## Key Benefits

- **Simple**: ~1,700 lines of focused code
- **Robust**: Handles edge cases from Arena 10K analysis  
- **Unbiased**: Proper calibration and cross-fitting
- **Practical**: Clear diagnostics for debugging

## References

Based on the paper: "Causal Judge Evaluation (CJE): Unbiased, Calibrated & Cost-Efficient Off-Policy Metrics for LLM Systems"