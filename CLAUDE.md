# CLAUDE.md

Core guidance for Claude Code when working with the CJE repository.

## 🎯 Project Philosophy

The `cje_simplified/` directory is a clean reimplementation focusing on:
- Clear separation of concerns
- Type safety with Pydantic models  
- Explicit error handling (no magic fallbacks)
- Simple, composable abstractions
- YAGNI (You Aren't Gonna Need It) - avoid premature abstraction

## 📁 Repository Structure

```
cje_simplified/           # ALL NEW WORK GOES HERE
├── calibration/          # Calibration utilities
├── core/                 # Core abstractions
├── data/                 # Data models and loading  
├── teacher_forcing/      # Log probability computation
├── experiments/          # Arena experiment pipeline
└── tests/                # Test suite
```

## 🚀 Quick Start

```python
from cje_simplified import load_dataset_from_jsonl, calibrate_dataset, PrecomputedSampler, CalibratedIPS

# Load data (no rewards required)
dataset = load_dataset_from_jsonl("data.jsonl")

# Calibrate if needed
calibrated_dataset, result = calibrate_dataset(
    dataset, 
    judge_field="judge_score",
    oracle_field="oracle_label"
)

# Run estimation
sampler = PrecomputedSampler(calibrated_dataset)
estimator = CalibratedIPS(sampler)
results = estimator.fit_and_estimate()
```

## 🔧 Essential Commands

```bash
# Run tests
poetry run pytest cje_simplified/

# Run experiments
cd cje_simplified/experiments/arena_10k_simplified

# Step 1: Generate data (no calibration)
poetry run python generate_arena_data.py --n-samples 1000

# Step 2: Analyze with calibration
poetry run python analyze_dataset.py --data data/cje_dataset.jsonl --oracle-coverage 0.5
```

## 🔑 API Keys

Required keys:
- `OPENAI_API_KEY` - For judge and oracle evaluation
- `FIREWORKS_API_KEY` - For response generation and log probabilities

```bash
source /Users/eddielandesberg/PycharmProjects/causal-judge-evaluation/set_secrets.sh
```

## 🧾 Command Best Practices

- Run "source set_secrets.sh" in the same line as other commands when they depend on api keys

## 📊 Data Format

Expected JSONL format:
```json
{
  "prompt": "What is 2+2?",
  "response": "4",
  "base_policy_logprob": -35.704,
  "target_policy_logprobs": {
    "pi_improved": -32.123
  },
  "metadata": {
    "judge_score": 0.8,
    "oracle_label": 0.85
  }
}
```

Note: `reward` field is added during analysis, not data generation.

## 🤖 Template Handling

For Fireworks models, templates are auto-detected:
```python
# Just pass None for template_config
result = compute_chat_logprob(chat, model)  # Auto-detects for Fireworks
```

Don't create complex abstractions for template selection - let the tools handle it.

## 🏗️ Key Architectural Decisions

1. **Clean Separation**: Data generation vs analysis are separate steps
2. **Optional Rewards**: Datasets can exist without rewards  
3. **Explicit Failures**: Use `None` for failures, never magic values
4. **Metadata Collection**: Non-core fields go in metadata automatically
5. **Transparent Filtering**: Use `sampler.n_valid_samples` to see samples after filtering
6. **Weight Calibration**: Variance can increase when uniform weights need structure (this is correct)

## ⚠️ Common Pitfalls

1. **Wrong field names**: Use `base_policy_logprob`, not `p0_logprob`
2. **Magic fallbacks**: Never use -100.0 or similar as fallbacks
3. **Mixing concerns**: Calibration happens in analysis, not data prep
4. **Assuming rewards exist**: Check before using PrecomputedSampler

## 🚨 Red Flags in Code Review

- Imports from `cje/` instead of `cje_simplified/`
- Magic numbers as fallbacks
- Classes with multiple responsibilities
- Calibration during data generation
- Unnecessary abstractions (if it's only used once, inline it)
- String-based dispatch when direct calls would suffice

## 🎨 Design Principles

1. **YAGNI (You Aren't Gonna Need It)**
   - Don't create abstractions for single use cases
   - Inline code that's only called from one place
   - Remove layers that don't add value

2. **Explicit is Better than Implicit**
   - No magic strings or hidden behavior
   - Clear function signatures and return types
   - Obvious data flow

3. **Fail Fast and Clearly**
   - Return None or raise exceptions, never magic values
   - Helpful error messages that guide users
   - Don't hide failures

Remember: The goal is to be **simple, correct, and maintainable**.