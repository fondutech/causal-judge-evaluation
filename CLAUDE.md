# CLAUDE.md

Core guidance for Claude Code when working with the CJE repository.

## 🎯 Project Philosophy

The `cje/` directory is the production implementation focusing on:
- Clear separation of concerns
- Type safety with Pydantic models  
- Explicit error handling (no magic fallbacks)
- Simple, composable abstractions
- YAGNI (You Aren't Gonna Need It) - avoid premature abstraction

## 📝 Documentation Principles

- Keep documentation minimal and focused on core concepts
- Avoid adding implementation details that will become outdated
- Focus on principles and patterns rather than specific code
- Update README.md for user-facing changes, keep CLAUDE.md for timeless guidance

## 📁 Repository Structure

```
cje/                      # Production implementation
├── calibration/          # Calibration utilities (isotonic, judge calibration)
├── data/                 # Data models, loading, validation
├── estimators/           # IPS, DR, MRDR, TMLE estimators
├── utils/                # Utilities (diagnostics, export, fresh draws)
├── visualization/        # Plotting and dashboard generation
├── teacher_forcing/      # Log probability computation
├── experiments/          # Arena experiment pipeline
└── tests/                # Comprehensive test suite
```

## 🚀 Quick Start

```python
from cje import load_dataset_from_jsonl, calibrate_dataset, PrecomputedSampler, CalibratedIPS

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
poetry run pytest cje/

# Run experiments
cd cje/experiments/arena_10k_simplified

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
6. **Stacked Weight Calibration**: SIMCal combines multiple candidates to minimize OOF variance
7. **Three Isotonic Mappings**: Global f_all for rewards, cross-fitted f^(-k) for DR, stacked SIMCal for weights
8. **DR via Inheritance**: DR inherits from CalibratedIPS to reuse weight machinery
9. **Mandatory prompt_id**: Required for DR to align logged data with fresh draws
10. **Fold ID Remapping**: Automatic remapping to [0..K-1] for subset compatibility

## ⚠️ Common Pitfalls

1. **Wrong field names**: Use `base_policy_logprob`, not `p0_logprob`
2. **Magic fallbacks**: Never use -100.0 or similar as fallbacks
3. **Mixing concerns**: Calibration happens in analysis, not data prep
4. **Assuming rewards exist**: Check before using PrecomputedSampler

## 🚨 Red Flags in Code Review

- Imports from old legacy paths
- Magic numbers as fallbacks
- Classes with multiple responsibilities
- Calibration during data generation
- Unnecessary abstractions (if it's only used once, inline it)
- String-based dispatch when direct calls would suffice

## 🔬 Three Isotonic Mappings

The codebase implements three distinct isotonic regressions, each with a specific purpose:

**Important Update (Aug 2024)**: Weight calibration now correctly uses judge scores as the ordering index by default (as specified in CJE paper Section 2.2). Ties in judge scores are handled by pooling weights within tied groups before applying PAV.

1. **Reward Calibration** (judge → oracle)
   - **Where**: `JudgeCalibrator` in `calibration/judge.py`
   - **Global model**: `f_all` for stable reward labels (`Sample.reward`)
   - **Cross-fitted**: `f^(-k)` for DR outcome models (via `predict_oof`)
   - **Usage**: `calibrate_dataset(enable_cross_fit=True)` for DR

2. **Weight Calibration** (IPS stabilization via stacked SIMCal)
   - **Where**: `SIMCalibrator` in `calibration/simcal.py`
   - **Method**: Stacks {baseline, increasing, decreasing} via OOF variance minimization
   - **Automatic**: Uses DR residuals when calibrator available, else IPS rewards
   - **No cross-fitting**: Applied per-policy in `CalibratedIPS.fit()`
   - **Purpose**: Prevents weight explosion while preserving mean

3. **DR Outcome Model** (g(s))
   - **Preferred**: `CalibratorBackedOutcomeModel` - reuses f^(-k) from calibration
   - **Fallback**: `IsotonicOutcomeModel` - refits if no calibrator passed
   - **Always cross-fitted**: Preserves orthogonality for DR

## 🤖 Doubly Robust (DR) Design

### Architecture
- **DR inherits from CalibratedIPS**: Reuses all weight machinery
- **Outcome models are composed**: Easy to swap different models
- **Cross-fitting in outcome models**: Each model handles its own k-fold logic
- **Fresh draws are separate**: Added via `add_fresh_draws()` after creation
- **CalibratorBackedOutcomeModel**: Reuses cross-fitted calibrators from reward calibration

### Key Classes
```python
# Estimator hierarchy
DREstimator(CalibratedIPS)  # Base DR with IPS correction
├── DRCPOEstimator          # Default with isotonic outcome model
├── MRDREstimator           # Policy-specific weighted outcome models
└── TMLEEstimator           # Targeted minimum loss estimation

# Outcome model hierarchy  
BaseOutcomeModel (ABC)            # Handles cross-fitting infrastructure
├── IsotonicOutcomeModel         # Default: g(x,a,s) = f(s)
├── CalibratorBackedOutcomeModel # Reuses calibrator's f^(-k) models
└── LinearOutcomeModel            # Example custom model
```

### Cross-Fitting for DR
```python
# ALWAYS enable cross-fitting and pass calibrator for DR
calibrated_dataset, result = calibrate_dataset(
    dataset,
    enable_cross_fit=True,  # Fits both f_all and f^(-k)
    n_folds=5
)

# ALWAYS pass calibrator to avoid redundant fitting
sampler = PrecomputedSampler(calibrated_dataset)
dr = DRCPOEstimator(sampler, calibrator=result.calibrator, n_folds=5)
# This reuses the cross-fitted models from calibration (efficient!)
```

### Implementation Pattern
Users only implement single-model logic:
```python
class CustomOutcomeModel(BaseOutcomeModel):
    def _fit_single_model(self, prompts, responses, rewards, judge_scores):
        # Train one model
        
    def _predict_single_model(self, model, prompts, responses, judge_scores):
        # Predict with one model
```

The base class handles all cross-fitting complexity.

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