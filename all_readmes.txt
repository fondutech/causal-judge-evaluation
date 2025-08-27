# CJE Project Documentation - All README Files
Generated on: Wed Aug 27 14:02:09 PDT 2025
=


# ============================================================================
# FILE: README.md
# ============================================================================

# Causal Judge Evaluation (CJE) with SIMCal

[![Tests](https://img.shields.io/badge/tests-155%20passing-brightgreen)](cje/tests)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Shape-Constrained, Unbiased Off-Policy Metrics for LLM Systems and Beyond**

CJE transforms routine LLM evaluation logs into unbiased, variance-controlled estimates of counterfactual performance: *"What would our KPI be if we shipped policy π′ instead of π₀?"*

## 🎯 The Problem

Modern LLM evaluation relies on automatic judges (GPT-4, Claude, etc.) to score outputs at scale. But these offline metrics are **correlational**—computed under your logging policy π₀, they don't answer the **causal** question of how a new policy π′ would perform if deployed.

## 💡 The Solution: SIMCal

CJE recasts judge-based evaluation as **calibrated causal inference** using our novel **Surrogate-Indexed Monotone Calibration (SIMCal)** - providing unbiased estimates with explicit variance control.

## 🚀 Quick Start

```bash
# Install from source
git clone https://github.com/fondutech/causal-judge-evaluation.git
cd causal-judge-evaluation
pip install -e .  # Install in development mode

# Set API key for log probability computation
export FIREWORKS_API_KEY="your-api-key"
```

```python
from cje import analyze_dataset

# One-line causal evaluation
results = analyze_dataset(
    "logs.jsonl",
    estimator="calibrated-ips"  # Uses SIMCal by default
    # Automatically uses all available oracle labels for calibration
)

print(f"Policy value: {results.estimates[0]:.3f} ± {1.96 * results.standard_errors[0]:.3f}")
```

## 📊 How It Works

CJE transforms biased judge scores into unbiased policy estimates through a principled pipeline:

```
┌─────────────────────────────────────────────────────────────────────┐
│                            INPUT DATA                               │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Logged Conversations                     │    │
│  │  • Prompts (X)                                              │    │
│  │  • Responses (A) from policy π₀                             │    │
│  │  • Judge scores (S) from automatic evaluator                │    │
│  │  • Log probabilities: log p_π₀(A|X)                         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                │                                    │
│          ┌─────────────────────┴──────────────────────┐             │
│          ▼                                            ▼             │
│  ┌───────────────┐                          ┌────────────────────┐  │
│  │ Oracle Subset │                          │ Full Dataset       │  │
│  │ (~5-10% data) │                          │ (100% of data)     │  │
│  │ + Human labels│                          │ Judge scores only  │  │
│  │     (Y)       │                          │                    │  │
│  └───────────────┘                          └────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
           │                                            │
           ▼                                            │
  ┌──────────────────────────┐                          │
  │   CALIBRATION STEP       │                          │
  │                          │                          │
  │ Learn f: S → Y via       │                          │
  │ isotonic regression      │                          │
  │ • Cross-fit with k folds │                          │
  │ • f^(-k) for fold k      │                          │
  └──────────────────────────┘                          │
           │                                            │
           └──────────────┬─────────────────────────────┘
                          ▼
                ┌─────────────────────┐        ┌──────────────────────┐
                │ Apply Calibration   │        │ Compute Importance   │
                │                     │        │      Weights         │
                │ R_i = f^(-k_i)(S_i) │        │                      │
                │ for ALL samples     │  ────► │ W_i = p_π′(A_i|X_i)  │
                │ using cross-fitted  │        │       ────────────   │
                │ models              │        │       p_π₀(A_i|X_i)  │
                └─────────────────────┘        └──────────────────────┘
                          │                              │
                          ▼                              ▼
                ┌─────────────────────┐        ┌──────────────────────┐
                │ Calibrated Rewards  │        │  SIMCal Projection   │
                │                     │        │                      │
                │ • Unbiased under    │        │ W_c = Proj_𝒮(W)      │
                │   monotonicity      │        │                      │
                │ • All samples have  │        │ • 𝒮 = monotone fns   │
                │   calibrated R      │        │   indexed by S       │
                └─────────────────────┘        │ • Mean preserving    │
                          │                    │ • Var(W_c) ≤ Var(W)  │
                          │                    └──────────────────────┘
                          │                              │
                          └──────────┬───────────────────┘
                                     ▼
                          ┌──────────────────────┐
                          │  Oracle Augmentation │
                          │  (Auto when < 100%)  │
                          │                      │
                          │ AUG = (L/p) × m̂(S) × │
                          │      (Y - f̂(S))      │
                          │                      │
                          │ • Accounts for       │
                          │   calibration        │
                          │   uncertainty        │
                          │ • Honest CIs         │
                          └──────────────────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │   FINAL ESTIMATE     │
                          │                      │
                          │  IPS Estimator:      │
                          │  V̂ = Σ(W_c,i×R_i     │
                          │      + AUG_i)/n      │
                          │                      │
                          │  DR (if fresh draws):│
                          │  V̂ = ĝ + Σ(W_c,i ×   │
                          │      (R_i - q̂_i)     │
                          │      + AUG_i)/n      │
                          │                      │
                          │  SE = std(ψ̂)/√n      │
                          │  95% CI: V̂ ± 1.96×SE │
                          └──────────────────────┘

Legend: X=prompts, A=responses, S=judge scores, Y=oracle labels, R=rewards
        W=importance weights, W_c=calibrated weights, ψ̂=influence function
        L=oracle label indicator, p=oracle coverage, m̂(S)=E[W|S], AUG=augmentation
```

### Key Innovation: SIMCal

SIMCal prevents weight explosion by projecting importance weights onto monotone functions indexed by the judge score, ensuring:
- ✅ Mean preservation (unbiasedness)
- ✅ Variance reduction via majorization
- ✅ Explicit variance cap for stability
- ✅ Oracle slice augmentation for honest CIs that account for calibration uncertainty

## 💻 Example: Comparing Model Versions

```python
from cje import analyze_dataset

# Evaluate if switching from Llama-3-8B to Llama-3-70B improves quality
# Note: Your data should have target_policy_logprobs for the policies you want to evaluate
results = analyze_dataset(
    "llama_logs.jsonl",  # Your logged Llama-3-8B conversations with judge scores
    estimator="calibrated-ips"
    # Uses available oracle labels (if any) for calibration
)

# Compare policies (assumes your data has these in target_policy_logprobs)
target_policies = results.metadata.get("target_policies", [])
for i, policy in enumerate(target_policies):
    estimate = results.estimates[i]
    stderr = results.standard_errors[i]
    print(f"{policy}: {estimate:.3f} ± {1.96*stderr:.3f}")
    
# Check diagnostics
if results.diagnostics and results.diagnostics.weight_ess < 0.1:
    print("⚠️ Warning: Low effective sample size - consider more data")
```

## 📋 Data Format

CJE expects JSONL logs with:
```json
{
  "prompt": "What is machine learning?",
  "response": "Machine learning is...",
  "base_policy_logprob": -35.704,  // Log P(response|prompt) under π₀
  "target_policy_logprobs": {       // Optional: pre-computed for efficiency
    "policy_a": -32.456,
    "policy_b": -33.789
  },
  "metadata": {
    "judge_score": 0.85,      // Required: automatic judge score
    "oracle_label": 0.90       // Optional: ground truth (for calibration slice)
  }
}
```

**Note**: For DR estimators, you'll also need fresh draws (new responses from π′ evaluated by the judge). These should be provided via the `fresh_draws_dir` parameter or by calling `estimator.add_fresh_draws()`. See the experiments in `cje/experiments/arena_10k_simplified/ablations/` for examples.

## 🔬 Available Estimators

| Estimator | Description | When to Use |
|-----------|-------------|-------------|
| **CalibratedIPS** | IPS with SIMCal weight calibration | Logged data only; best IPS variant |
| **RawIPS** | Standard importance sampling | Baseline comparison; diagnostic purposes |
| **StackedDREstimator** | Optimal combination of DR methods | Fresh draws; **recommended DR default** |
| **DRCPOEstimator** | Doubly-Robust Counterfactual Policy Optimization | Fresh draws; basic DR method |
| **MRDREstimator** | More Robust Doubly-Robust estimator | Fresh draws + concern about misspecification |
| **TMLEEstimator** | Targeted maximum likelihood | Fresh draws + want optimal efficiency |

## 🔍 Diagnostics & Quality Gates

CJE provides comprehensive diagnostics to audit assumptions:

```python
import warnings

diagnostics = results.diagnostics
print(diagnostics.summary())

# Automatic quality gates
if diagnostics and diagnostics.weight_ess < 0.1:  # Less than 10% effective sample size
    warnings.warn("Low ESS - consider tighter variance cap")
if diagnostics and diagnostics.tail_indices:
    worst_tail = min(diagnostics.tail_indices.values())
    if worst_tail < 2:
        warnings.warn("Heavy tails detected - results may be unstable")
```

Key diagnostics include:
- **ESS (Effective Sample Size)**: Overlap quality metric
- **Tail Index**: Heavy tail detection via Hill estimator
- **Calibration Curves**: Judge reliability visualization
- **Orthogonality Scores**: DR assumption checking

## 🛠️ Advanced Usage

### Multiple Estimators
```python
# Run multiple estimators for robustness
for estimator in ["calibrated-ips", "stacked-dr"]:
    results = analyze_dataset("logs.jsonl", estimator=estimator)
    print(f"{estimator}: {results.estimates[0]:.3f}")
```

### Custom Calibration
```python
from cje import load_dataset_from_jsonl, calibrate_dataset, PrecomputedSampler, CalibratedIPS

# Fine-grained control
dataset = load_dataset_from_jsonl("logs.jsonl")
calibrated_dataset, cal_result = calibrate_dataset(
    dataset,
    judge_field="judge_score",
    oracle_field="oracle_label"
    # Uses all available oracle labels for calibration
)

sampler = PrecomputedSampler(calibrated_dataset)
estimator = CalibratedIPS(sampler, var_cap=0.5)  # Tighter variance cap
results = estimator.fit_and_estimate()
```

### Oracle Slice Augmentation (Automatic)
```python
# Automatic: CJE detects partial oracle coverage and enables augmentation
estimator = CalibratedIPS(sampler)  # Auto-enables if 0% < oracle coverage < 100%
results = estimator.fit_and_estimate()

# Check if augmentation was applied (for each policy)
if "slice_augmentation" in results.metadata:
    for policy, aug_info in results.metadata["slice_augmentation"].items():
        if "slice_variance_share" in aug_info:
            print(f"{policy} oracle uncertainty: {aug_info['slice_variance_share']:.1%} of total variance")

# Manual control if needed
estimator = CalibratedIPS(sampler, oracle_slice_config=False)  # Disable
estimator = CalibratedIPS(sampler, oracle_slice_config=True)   # Force enable
```

### CF-bits: Information Accounting for Reliability
```python
from cje.cfbits import cfbits_report_logging_only

# Get comprehensive reliability analysis
report = cfbits_report_logging_only(estimator, "target_policy")

# Check if estimate is trustworthy
if report["gates"]["state"] == "REFUSE":
    print("Cannot trust this estimate - catastrophic overlap")
elif report["gates"]["state"] == "CRITICAL":
    print(f"Use with extreme caution: {report['gates']['reasons']}")

# View uncertainty decomposition
print(f"A-ESSF: {report['overlap']['aessf']:.1%}")  # Structural ceiling
print(f"Sampling width: {report['sampling_width']['wvar']:.3f}")
print(f"Total uncertainty: {report['cfbits']['w_tot']:.3f}")
```

CF-bits decomposes uncertainty into identification width (structural limits) and sampling width (statistical noise), providing actionable reliability gates.

## 📚 Documentation

- **[Sphinx Documentation](docs/)**: Complete API reference and theory guides
- **[Example Usage](cje/example_usage.py)**: Python examples demonstrating all workflows
- **[Arena Experiment](cje/experiments/arena_10k_simplified/)**: Complete real-world ablation study with 13.9× ESS improvement

### Module Documentation
- **[Calibration](cje/calibration/)**: SIMCal algorithm and isotonic regression
- **[Data](cje/data/)**: Data models and validation
- **[Teacher Forcing](cje/teacher_forcing/)**: Log probability computation
- **[Estimators](cje/estimators/)**: All estimation methods
- **[CF-bits](cje/cfbits/)**: Information accounting for uncertainty decomposition

## 🧪 Testing

```bash
# Run all tests
poetry run pytest cje/tests/

# With coverage
poetry run pytest --cov=cje --cov-report=html
```

## 🔑 Key Concepts

- **π₀**: Your logging/base policy (what generated the data)
- **π′**: Target policy you want to evaluate
- **Judge Score (S)**: Automatic evaluation score (e.g., from GPT-4)
- **Oracle Label (Y)**: Human ground truth for calibration
- **SIMCal**: Our variance reduction method via monotone projection
- **ESS**: Effective Sample Size - measures overlap quality

## 📖 Citation

If you use CJE in your research, please cite:
```bibtex
@software{cje2025,
  title={Causal Judge Evaluation (CJE): A Framework for Unbiased Off-Policy 
         Evaluation with Shape-Constrained Calibration},
  author={CJE Contributors},
  year={2025},
  url={https://github.com/fondutech/causal-judge-evaluation}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

This work builds on foundational research in off-policy evaluation (Horvitz-Thompson, Dudík et al.), isotonic calibration (van der Laan et al. 2025), and semiparametric efficiency (Bickel et al., Chernozhukov et al.).

---

**Made with ❤️ for better offline evaluation**

# ============================================================================
# FILE: cje/.pytest_cache/README.md
# ============================================================================

# pytest cache directory #

This directory contains data from the pytest's cache plugin,
which provides the `--lf` and `--ff` options, as well as the `cache` fixture.

**Do not** commit this to version control.

See [the docs](https://docs.pytest.org/en/stable/how-to/cache.html) for more information.


# ============================================================================
# FILE: cje/calibration/README.md
# ============================================================================

# CJE Calibration Module

## Overview

The calibration module implements the core mathematical machinery that enables unbiased causal inference from judge-based evaluations. It provides three distinct calibration approaches that work together to transform raw logged data into reliable policy value estimates with controlled variance.

## When to Use Each Calibration

### Use **Reward Calibration** when:
- You have judge scores and some oracle labels
- You want to map judge scores → oracle scale
- You're using any estimation method

### Use **Weight Calibration** (SIMCal) when:
- Importance weights have high variance
- You want to stabilize IPS estimates
- You're using CalibratedIPS estimator

### Use **Cross-Fitted Models** when:
- You're using DR estimators
- You need orthogonality guarantees
- You have enough data for stable folds

## File Structure

```
calibration/
├── __init__.py          # Public API exports
├── dataset.py           # High-level dataset calibration workflows
├── flexible_calibrator.py # Flexible calibration for non-monotone relationships
├── isotonic.py          # Core isotonic regression and variance control
├── judge.py             # Judge score calibration to oracle labels
├── oracle_slice.py      # Oracle slice uncertainty augmentation
├── simcal.py            # Stacked SIMCal implementation
└── iic.py               # Isotonic Influence Control for variance reduction
```

## Core Concepts

### 1. Judge Score Calibration
Maps cheap LLM judge scores to expensive oracle labels. Default is 'auto' mode which automatically selects between:
- **Monotone calibration**: Standard isotonic regression (when relationship is monotone)
- **Flexible calibration**: Two-stage g(S)→isotonic for non-monotone relationships

Auto mode detects non-monotonicity by comparing regional performance and selects the appropriate method. The selected mode is stored in metadata for transparency.

### 2. Weight Calibration (SIMCal)
Stabilizes importance weights through score-indexed monotone projection:
- Projects weights to be monotone with an ordering index
- Enforces variance constraints via blending
- Maintains mean-1 property for unbiasedness

### 3. Cross-Fitted Models
For doubly robust methods, provides out-of-fold predictions to maintain orthogonality between nuisance functions.

### 4. Oracle Slice Augmentation
When we calibrate judge scores using only a subset of oracle labels (e.g., 10% coverage), the calibration function f̂ itself has uncertainty. Standard IPS/DR methods treat f̂ as fixed, leading to overconfident CIs. Oracle slice augmentation corrects this by adding a term that accounts for calibration uncertainty, ensuring CIs properly widen when oracle coverage is low.

### 5. Isotonic Influence Control (IIC)
A variance reduction technique that residualizes influence functions against judge scores. By fitting E[φ|S] using isotonic regression and computing residuals φ̃ = φ - Ê[φ|S], IIC reduces variance without changing the estimand. This is "free" variance reduction that's enabled by default in all estimators.

## Module Descriptions

### `dataset.py` - Dataset Calibration Workflows
High-level functions that orchestrate the calibration process for entire datasets:
- `calibrate_dataset()`: Transforms Dataset objects with judge scores into calibrated rewards
- `calibrate_from_raw_data()`: Works with raw dictionaries for pipeline integration
- Handles both standard and cross-fitted calibration
- Preserves metadata and adds calibration diagnostics

### `judge.py` - Judge Calibration
Implements calibration from judge scores to oracle labels with auto mode selection:
- `JudgeCalibrator`: Main calibration class with flexible mode support
- `fit_transform()`: Standard calibration on oracle subset
- `fit_cv()`: Cross-fitted calibration for DR methods
- `index()`: Returns transformation for outcome models (S for monotone, g(S) for two-stage)
- `CalibrationResult`: Container for calibrated scores and diagnostics
- Auto mode (default): Automatically selects monotone or flexible calibration
- Supports partial labeling (oracle coverage)

### `flexible_calibrator.py` - Non-Monotone Calibration
Handles non-monotone judge→oracle relationships via two-stage approach:
- `FlexibleCalibrator`: Implements g(S)→isotonic calibration
- First stage: Learn smooth transformation g(S) using splines
- Second stage: Apply isotonic regression on g(S)
- `index()`: Exposes the transformation T=g(S) for outcome models
- Per-fold ECDF for consistent rank transformation
- Auto selection based on regional performance comparison

**Mode Selection Logic:**
- Compares monotone vs two-stage using 1-SE rule
- Checks performance across score regions (low/mid/high)
- Selects two-stage if better in ≥2/3 regions or significantly better overall
- Optimized to skip two-stage training when monotone is clearly sufficient

**Technical Details:**
- ECDF-based ranking prevents distribution leakage between folds
- Minimum 5 spline knots to avoid underfitting
- Fallback to monotone for small samples (<20)
- Clipping to [0,1] ensures valid reward range

### `isotonic.py` - Isotonic Weight Calibration
Core mathematical operations for weight calibration:
- `calibrate_to_target_mean()`: Main entry point for weight calibration
- `_pav_mean1_projection_sorted()`: Pool Adjacent Violators with mean preservation
- `_variance_safe_blend_closed_form()`: Optimal blending for variance control
- Uses "exact" mode (bisection) for consistency
- Handles ordering by arbitrary index (e.g., judge scores)

### `simcal.py` - Stacked SIMCal
Advanced weight calibration through stacking:
- `SIMCalibrator`: Combines {baseline, increasing, decreasing} candidates
- Out-of-fold (OOF) influence function minimization
- Quadratic program on simplex for optimal mixture
- Uniform blending for ESS/variance constraints
- Configurable via `SimcalConfig` dataclass

### `oracle_slice.py` - Oracle Slice Augmentation
Corrects for bias and variance when using calibrated rewards instead of true oracle labels:
- **Problem**: We use f̂(S) everywhere but only have true Y on oracle subset
- **Solution**: Add augmentation term `(L/p) * m̂(S) * (Y - f̂(S))` where:
  - L indicates oracle label presence, p = oracle coverage
  - m̂(S) = E[W|S] estimated via isotonic regression
  - Unbiased for the gap between proxy and truth
- **Effect**: Wider, honest CIs that reflect calibration uncertainty
- Auto-enables when 0% < oracle coverage < 100%

### `iic.py` - Isotonic Influence Control
Reduces influence function variance through residualization:
- `IsotonicInfluenceControl`: Main class for applying IIC
- Fits E[φ|S] using isotonic regression (with optional cross-fitting)
- **Automatic direction selection**: Uses Spearman correlation to choose increasing/decreasing
- Returns residuals φ̃ = φ - Ê[φ|S] with reduced variance
- Enabled by default in all estimators (use_iic=True)
- Provides diagnostics: R², variance reduction, ESS gain
- Typical variance reductions: 20-95% depending on policy correlation with judge scores
- Key insight: Influence functions often correlate with judge scores, so removing the predictable component reduces variance "for free"

## Key Design Decisions

### 1. **Separation of Concerns**
Each calibration type is isolated with clear interfaces:
- Reward calibration doesn't know about weights
- Weight calibration doesn't know about rewards
- Outcome models are separate from both

### 2. **Mean Preservation**
All calibrations preserve population means exactly:
- Isotonic regression preserves E[Y]
- Weight calibration maintains mean = 1 (for Hájek normalization)
- Critical for unbiased estimation

### 3. **Variance Control**
Multiple mechanisms for variance reduction:
- **Isotonic projection**: Can reduce variance when weights correlate with ordering index
- **Variance cap**: Explicit upper bound on weight variance via blending
- **ESS floor**: Minimum effective sample size constraint
- **Baseline shrinkage**: Small bias for large variance reduction

### 4. **Cross-Fitting Support**
Built-in support for cross-fitted calibration:
- Prevents overfitting in DR methods
- Maintains orthogonality between nuisance functions
- Uses unified fold system from `cje.data.folds` for consistency
- Fold assignments computed deterministically from prompt_id

### 5. **Numerical Robustness**
Careful handling of edge cases:
- Zero weights: Fallback to uniform
- Constant weights: Return target mean
- Sparse weights: Relaxed tolerance
- Numerical precision: Multiple safety checks


## Mathematical Foundations

### Isotonic Regression (PAV Algorithm)
Finds the best-fitting monotone function: `min ||f(x) - y||²` subject to monotonicity.
- **Time**: O(n log n) 
- **Property**: When ordered by uncorrelated index, produces nearly constant weights

### Mean-Preserving Projection  
Ensures calibrated weights have exactly mean=1 via bisection on Lagrange multipliers.
- **Why**: Critical for unbiased estimation (E[W] = 1)
- **Implementation**: ~30-40 PAV calls for exact solution

### Variance-Safe Blending
Optimally blends raw and calibrated weights to satisfy variance constraints:
```
w_final = (1-α)·raw + α·calibrated
where Var(w_final) ≤ ρ·Var(raw)
```
- **Solution**: Closed-form via quadratic formula

### Stacked SIMCal
Combines K=3 candidates by minimizing OOF influence variance:
```
min_π π'Σπ s.t. π ≥ 0, Σπ = 1
```
- **Candidates**: {baseline, increasing, decreasing}
- **Solution**: Quadratic program on simplex

## Usage Patterns

### Basic Reward Calibration
```python
from cje.calibration import calibrate_dataset

# Calibrate judge scores to oracle labels (auto mode by default)
calibrated_dataset, cal_result = calibrate_dataset(
    dataset,
    judge_field="judge_score",
    oracle_field="oracle_label",
    calibration_mode="auto",  # Or "monotone", "two_stage"
    random_seed=42  # For reproducibility
)

# Access calibration quality metrics and metadata
print(f"RMSE: {cal_result.calibration_rmse:.3f}")
print(f"Coverage: {cal_result.coverage_at_01:.1%}")
print(f"Selected mode: {calibrated_dataset.metadata.get('calibration_info', {}).get('selected_mode')}")
```

### Weight Calibration (Direct)
```python
from cje.calibration import calibrate_to_target_mean

# Calibrate weights with variance control
calibrated_weights, info = calibrate_to_target_mean(
    raw_weights,
    target_mean=1.0,
    enforce_variance_nonincrease=True,
    ordering_index=judge_scores,  # Order by judge scores
    return_diagnostics=True
)

print(f"Variance reduction: {info['var_before']/info['var_after']:.2f}x")
```

### Stacked SIMCal
```python
from cje.calibration import SIMCalibrator, SimcalConfig

# Configure stacked calibration
config = SimcalConfig(
    ess_floor=0.2,      # Minimum 20% ESS
    var_cap=1.0,        # No variance increase
    include_baseline=False,
)

# Run calibration
calibrator = SIMCalibrator(config)
calibrated, info = calibrator.transform(
    weights, 
    judge_scores,
    rewards=rewards  # For IPS influence functions
)

print(f"Mixture: {info['mixture_weights']}")
print(f"ESS improvement: {info['ess_after']/info['ess_before']:.2f}x")
```

### Cross-Fitted Calibration (for DR)
```python
from cje.calibration import JudgeCalibrator

# Fit with cross-validation for DR methods
calibrator = JudgeCalibrator()
result = calibrator.fit_cv(
    judge_scores,
    oracle_labels,
    oracle_mask,
    n_folds=5
)

# Get out-of-fold predictions
oof_predictions = calibrator.predict_oof(judge_scores, fold_ids)
```

### Oracle Slice Augmentation
```python
from cje.calibration import OracleSliceConfig
from cje import CalibratedIPS

# Automatic: Augmentation enables when 0% < oracle coverage < 100%
estimator = CalibratedIPS(sampler)  # Auto-detects and enables if needed
result = estimator.fit_and_estimate()

# Or configure explicitly if needed
oracle_config = OracleSliceConfig(
    enable_augmentation=True,
    enable_cross_fit=True,
    min_pi=0.01,  # Minimum labeling probability
    use_mar=False  # MCAR assumption for now
)

# Use with explicit configuration
estimator = CalibratedIPS(
    sampler,
    oracle_slice_config=oracle_config
)

# The augmentation automatically adjusts standard errors
# to account for calibration uncertainty
result = estimator.fit_and_estimate()

# Check slice contribution to variance (if augmentation was applied)
if "slice_augmentation" in result.metadata:
    aug_diag = result.metadata["slice_augmentation"]["policy_a"]
    print(f"Oracle slice variance share: {aug_diag['slice_variance_share']:.1%}")
```

## Configuration Options

### SimcalConfig Parameters
- `ess_floor`: Minimum ESS as fraction (e.g., 0.2 = 20%)
- `var_cap`: Maximum variance (e.g., 1.0 = no increase)
- `include_baseline`: Include raw weights in stack
- `baseline_shrink`: Shrinkage toward baseline (0-1)
- `ridge_lambda`: Ridge regularization for covariance
- `n_folds`: Number of OOF folds if not provided

### Calibration Modes
- **Auto** (default): Automatically selects between monotone and two-stage based on performance
- **Monotone**: Standard isotonic regression (forces monotone relationship)
- **Two-stage**: Flexible g(S)→isotonic for non-monotone relationships
- **Cross-fitted**: K-fold models for DR orthogonality (enable_cross_fit=True)
- **Projection mode**: Always uses "exact" (bisection) for consistency

## Implementation Details

### Ordering Index Flexibility
The `ordering_index` parameter in isotonic calibration allows weights to be monotone in any score:
- **None**: Sort by raw weights (backward compatibility)
- **Judge scores**: Align with human evaluation
- **Calibrated rewards**: Align with outcome models (for DR)

When the ordering index is uncorrelated with weights, isotonic projection produces nearly constant weights - this is expected and provides stabilization.

### Tie Handling
When the ordering index has ties (common with discrete judge scores):
1. Pool weights within tied groups (average)
2. Apply isotonic regression to pooled values
3. Assign same calibrated weight to all tied samples

### Numerical Tolerances
- `EPS = 1e-12`: Machine epsilon for comparisons
- `MEAN_TOL = 1e-10`: Tolerance for mean preservation
- `VAR_TOL = 1.001`: Allow 0.1% slack on variance cap

### Memory Efficiency
- Isotonic regression is O(n log n) time, O(n) space
- Stacked calibration builds K=3 candidates
- Cross-fitting stores K models but applies one at a time

## Common Issues and Solutions

### Issue: "Judge field 'reward' not allowed"
**Cause**: Trying to use 'reward' as judge field to avoid confusion  
**Solution**: Use a different field name in metadata (e.g., 'judge_score')

### Issue: Low calibration R² (< 0.3)
**Cause**: Judge scores poorly predict oracle labels  
**Solution**: 
- Increase oracle coverage (aim for >10%)
- Improve judge prompt/model
- Consider using a different judge
- Check if oracle labels are noisy

### Issue: Nearly constant calibrated weights
**Cause**: Ordering index uncorrelated with importance ratios  
**Solution**: This is expected and actually good - provides maximum variance stabilization

### Issue: Variance cap not satisfied exactly
**Cause**: Numerical precision or infeasible constraint  
**Solution**: Check info dict for 'feasible' flag and 'note' field

### Issue: ESS floor conflicts with variance cap
**Cause**: ESS implies tighter variance constraint than specified  
**Solution**: ESS constraint will dominate (warning issued)

### Issue: Very low oracle coverage (<5%)
**Cause**: Too few labeled samples for reliable calibration
**Solution**: 
- Collect more oracle labels
- Consider using judge scores directly (uncalibrated)
- Use bootstrapping to assess calibration uncertainty

## Testing

The calibration module has comprehensive test coverage:
- `test_stacked_simcal.py`: Stacked SIMCal functionality
- Integration tests verify calibration in full pipeline
- Edge case tests for degenerate inputs

Run tests:
```bash
poetry run pytest cje/tests/ -k calibration
```

## Performance Considerations

### Computational Complexity
- **Isotonic regression**: O(n log n) via PAV
- **Exact projection**: ~30-40 PAV calls (still O(n log n))
- **Stacked SIMCal**: O(n²K) for covariance (K=3 candidates)
- **Cross-fitting**: K × isotonic regression cost


### When to Use Each Method

**Use standard calibration when:**
- You have sufficient oracle labels (>100)
- Not using DR methods
- Speed is critical

**Use cross-fitted calibration when:**
- Using DR estimators
- Need orthogonality guarantees
- Have enough data for stable fold models

**Use stacked SIMCal when:**
- Weights have high variance
- Multiple candidate projections make sense
- OOF validation is feasible


## Advanced Topics

### Bootstrapping Calibration Uncertainty
```python
# For low oracle coverage scenarios
n_bootstrap = 100
calibrations = []
for _ in range(n_bootstrap):
    idx = np.random.choice(n_oracle, n_oracle, replace=True)
    cal = JudgeCalibrator()
    result = cal.fit_transform(judge_scores[idx], oracle_labels[idx])
    calibrations.append(result.calibrated_scores)
```

### Debugging SIMCal
```python
# Check intermediate steps
calibrated, info = calibrator.transform(weights, scores, rewards=rewards)
print(f"Mixture weights: {info['mixture_weights']}")
print(f"Variance reduction: {info['var_before']/info['var_after']:.2f}x")
```


## References

- **Isotonic Regression**: Robertson et al. (1988), "Order Restricted Statistical Inference"
- **PAV Algorithm**: Ayer et al. (1955), "An Empirical Distribution Function for Sampling with Incomplete Information"  
- **Majorization**: Marshall & Olkin (1979), "Inequalities: Theory of Majorization"
- **SIMCal**: CJE paper (2025), "Surrogate-Indexed Monotone Calibration"
- **Cross-fitting**: Chernozhukov et al. (2018), "Double/Debiased Machine Learning"

## Summary

The calibration module provides three essential transformations for causal inference: mapping judge scores to oracle labels, stabilizing importance weights through SIMCal, and enabling cross-fitted models for DR methods. Each calibration type maintains mean preservation for unbiased estimation while controlling variance through different mechanisms.

# ============================================================================
# FILE: cje/cfbits/README.md
# ============================================================================

# CF-bits: Information Accounting for Causal Inference

CF-bits provides an information-accounting layer that decomposes uncertainty into **identification width** (structural limits) and **sampling width** (statistical noise).

## Quick Start - Two Scenarios

### Scenario A: Fresh Draws Available (DR/TMLE)

```python
from cje.cfbits import cfbits_report_fresh_draws

# After fitting your DR/TMLE estimator with fresh draws
report = cfbits_report_fresh_draws(
    estimator, 
    policy="target_policy",
    n_boot=800,
    random_state=42
)

print(report["summary"])  # One-line summary
print(f"Gate: {report['gates']['state']}")  # GOOD/WARNING/CRITICAL/REFUSE
```

### Scenario B: Logging-Only (IPS/Cal-IPS)

```python
from cje.cfbits import cfbits_report_logging_only

# After fitting your IPS/CalibratedIPS estimator
report = cfbits_report_logging_only(
    estimator,
    policy="target_policy", 
    n_boot=800,
    random_state=42
)

if report["gates"]["state"] == "REFUSE":
    print("Cannot trust this estimate - need better overlap or fresh draws")
```

## Manual API (Full Control)

```python
from cje import load_dataset_from_jsonl, calibrate_dataset
from cje.data.precomputed_sampler import PrecomputedSampler
from cje.estimators.calibrated_ips import CalibratedIPS
from cje.cfbits import compute_ifr_aess, compute_sampling_width, estimate_overlap_floors

# Load and calibrate data
dataset = load_dataset_from_jsonl("data.jsonl")
calibrated_dataset, _ = calibrate_dataset(dataset, judge_field="judge_score")
sampler = PrecomputedSampler(calibrated_dataset)

# Fit estimator
estimator = CalibratedIPS(sampler)
result = estimator.fit_and_estimate()

# Compute CF-bits metrics
policy = "target_policy"

# 1. Sampling width (compute first to get var_oracle)
wvar, var_components = compute_sampling_width(estimator, policy)
print(f"Sampling width: {wvar:.3f}")

# 2. Efficiency metrics (IFR and aESS) using actual var_oracle
if_values = estimator.get_influence_functions(policy)
efficiency = compute_ifr_aess(if_values, var_oracle=var_components.var_oracle)
print(f"IFR: {efficiency.ifr_main:.2%}, aESS: {efficiency.aess_main:.1f}")
print(f"IFR (OUA): {efficiency.ifr_oua:.2%}, aESS (OUA): {efficiency.aess_oua:.1f}")

# 3. Overlap floors
weights = estimator.get_weights(policy) or sampler.compute_importance_weights(policy)
judge_scores = sampler.get_judge_scores()
overlap = estimate_overlap_floors(judge_scores, weights)
print(f"A-ESSF: {overlap.aessf:.2%}, BC: {overlap.bc:.2%}")
```

## Core Concepts

### Information Fraction Ratio (IFR)
- **What**: Ratio of efficient IF variance to actual IF variance
- **Two versions**:
  - `IFR_main`: Var(EIF)/Var(IF) - standard efficiency
  - `IFR_OUA`: Var(EIF)/(Var(IF) + n×Var_oracle) - accounts for oracle uncertainty
- **Range**: (0, 1] where 1 = perfectly efficient
- **Interpretation**: How close your estimator is to the efficiency bound

### Adjusted ESS (aESS)
- **What**: n × IFR
- **Two versions**: `aESS_main` and `aESS_OUA` corresponding to the IFR versions
- **Interpretation**: Equivalent sample size if you had an efficient estimator
- **Example**: n=1000, IFR=0.5 → aESS=500 (half as efficient as possible)

### A-ESSF (Adjusted ESS Fraction)
- **What**: Structural overlap on judge marginal σ(S)
- **Formula**: 1/(1 + χ²_S) where χ²_S = E[ω(S)²] - 1
- **Interpretation**: Upper bound on achievable efficiency via S-calibration

### CF-bits
- **Formula**: bits = log₂(W₀/W)
- **Interpretation**: Each bit represents a halving of uncertainty width
- **Decomposition**: bits_tot from (Wid + Wvar)

## Module Structure

```
cfbits/
├── core.py         # CF-bits computation and gates
├── sampling.py     # IFR, aESS, sampling width
├── overlap.py      # A-ESSF, BC on σ(S)
└── identification.py  # Wid computation (placeholder)
```

### Implemented Features
- ✅ IFR and aESS computation (with IFR_main/IFR_OUA distinction)
- ✅ Sampling width with oracle uncertainty augmentation (OUA via jackknife)
- ✅ Conservative A-ESSF and BC overlap metrics (no monotonicity assumption)
- ✅ CF-bits from widths
- ✅ Reliability gates with LCB support
- ✅ Ready-to-use playbooks for common scenarios

### Not Yet Implemented
- ⏳ Full identification width (Wid) with isotonic bands
- ⏳ Complete EIF for CalibratedIPS

## API Reference

### Efficiency Metrics

```python
compute_ifr_aess(phi, eif=None, n=None, var_oracle=0.0) → EfficiencyStats
```
Compute Information Fraction Ratio and adjusted ESS. Returns both IFR_main and IFR_OUA.

### Sampling Width

```python
compute_sampling_width(estimator, policy, alpha=0.05, use_iic=True, compute_oua=True) → (float, SamplingVariance)
```
Compute statistical uncertainty width with optional IIC variance reduction and oracle uncertainty.

### Overlap Metrics

```python
estimate_overlap_floors(S, W, method="conservative", n_boot=500) → OverlapFloors
```
Estimate structural overlap bounds on judge marginal. Uses conservative estimation without monotonicity assumptions.

### CF-bits

```python
compute_cfbits(w0, wid, wvar, ifr_main=None, ifr_oua=None) → CFBits
```
Compute bits of information from width components. Prefers IFR_OUA when available.

### Gates

```python
apply_gates(aessf=None, aessf_lcb=None, ifr=None, ifr_lcb=None, tail_index=None) → GatesDecision
```
Apply reliability thresholds and generate recommendations. Uses lower confidence bounds (LCBs) when available for conservative gating.

## Interpretation Guide

### IFR Values
- **> 0.8**: Excellent efficiency
- **0.5 - 0.8**: Good efficiency
- **0.2 - 0.5**: Moderate efficiency (consider improvements)
- **< 0.2**: Poor efficiency (investigate causes)

### A-ESSF Values
- **> 0.5**: Excellent structural overlap
- **0.2 - 0.5**: Good overlap
- **0.05 - 0.2**: Marginal (use DR methods)
- **< 0.05**: Catastrophic (consider different policies)

### CF-bits
- **0 bits**: No uncertainty reduction
- **1 bit**: Uncertainty halved
- **2 bits**: Uncertainty quartered
- **3+ bits**: Substantial reduction (8x or more)

## Mathematical Foundation

The key insight is that total uncertainty decomposes as:
```
W_tot = W_id + W_var
```

Where:
- **W_id**: What cannot be determined from logs (structural)
- **W_var**: Statistical noise from finite samples

This decomposition is auditable, with each component traceable to specific assumptions and data limitations.

## Troubleshooting

### "No influence functions available"
- Ensure estimator has been fitted: `estimator.fit_and_estimate()`
- Check that estimator stores IFs (all CJE estimators should)

### "IFR > 1.0"
- Numerical issue - IFR is theoretically bounded by 1
- Check for very small variances causing instability

### "A-ESSF > BC²"
- Theoretical violation suggesting numerical issues
- Increase bootstrap samples or check weight distribution

## References

- CF-bits paper: [Coming soon]
- CJE paper: [Calibrated Off-Policy Evaluation]
- Efficiency theory: Semiparametric efficiency bounds

# ============================================================================
# FILE: cje/data/README.md
# ============================================================================

# CJE Data Module

## Overview

The data module handles all data loading, validation, and preparation for CJE analysis. It provides type-safe data models using Pydantic, flexible data loading through factory patterns, and comprehensive validation to ensure data quality before estimation.

## When to Use

### Use **Dataset** when:
- You need a type-safe container for CJE data
- You're passing data between modules
- You want automatic validation

### Use **PrecomputedSampler** when:
- You have data with rewards ready for estimation
- You need importance weight computation
- You're feeding data to estimators

### Use **DatasetFactory** when:
- Loading data from JSONL files
- Converting raw dictionaries to typed Datasets
- You need flexible data loading patterns

### Use **FreshDrawDataset** when:
- You have fresh samples for DR estimation
- You need to organize per-policy fresh draws
- You're using DR/TMLE estimators

## File Structure

```
data/
├── __init__.py           # Public API exports
├── models.py             # Pydantic data models (Sample, Dataset, etc.)
├── loaders.py            # Data loading utilities (DatasetLoader, DataSource)
├── factory.py            # Factory pattern for Dataset creation
├── precomputed_sampler.py # Sampler wrapper for estimators
├── fresh_draws.py        # Fresh draw models for DR
├── folds.py              # Unified fold management for cross-validation
├── validation.py         # Data validation functions
└── reward_utils.py       # Reward manipulation utilities
```

## Core Concepts

### 1. Type-Safe Data Models
All data flows through Pydantic models with automatic validation:
- **Sample**: Single observation with prompt, response, rewards, and log probs
- **Dataset**: Collection of samples with target policies
- **EstimationResult**: Output from estimators with estimates and diagnostics

### 2. Factory Pattern
DatasetFactory provides a clean interface for loading data from various sources while maintaining flexibility through dependency injection.

### 3. Validation Layers
Data is validated at multiple levels:
- Pydantic field validation (types, ranges)
- Structural validation (required fields exist)
- Semantic validation (policies in data match declared targets)

## Common Interface

### Loading Data
```python
from cje.data import DatasetFactory

# From JSONL file
factory = DatasetFactory()
dataset = factory.create_from_jsonl("data.jsonl")

# From raw dictionaries
data = [{"prompt": "...", "response": "...", ...}, ...]
dataset = factory.create_from_data(data)
```

### Using PrecomputedSampler
```python
from cje.data import PrecomputedSampler

# Create sampler (requires rewards)
sampler = PrecomputedSampler(dataset)

# Or directly from JSONL
sampler = PrecomputedSampler.from_jsonl("calibrated_data.jsonl")

# Access data
n_samples = sampler.n_valid_samples
policies = sampler.target_policies

# Check oracle coverage (triggers automatic augmentation when < 1.0)
oracle_coverage = sampler.oracle_coverage  # Float in [0, 1]: fraction with oracle labels
```

### Data Validation
```python
from cje.data import validate_cje_data

# Check if data has required fields
is_valid, issues = validate_cje_data(
    data,
    judge_field="judge_score",
    oracle_field="oracle_label"
)
if not is_valid:
    for issue in issues:
        print(f"Issue: {issue}")
```

## Data Format

### Required Fields

Every sample must have:
- `prompt_id`: Unique identifier (checked in top-level, then metadata, auto-generated from prompt hash if missing)
- `prompt`: Input text/context
- `response`: Generated output
- `base_policy_logprob`: Log probability under logging policy
- `target_policy_logprobs`: Dict of log probs for target policies

### Optional Fields
- `reward`: Calibrated reward in [0, 1] (required for PrecomputedSampler)
- `metadata`: Dict containing additional fields like:
  - `judge_score`: Raw judge evaluation
  - `oracle_label`: Ground truth label

### Example JSONL Entry
```json
{
  "prompt_id": "example_001",
  "prompt": "What is machine learning?",
  "response": "Machine learning is a subset of AI...",
  "base_policy_logprob": -45.67,
  "target_policy_logprobs": {
    "gpt4": -42.31,
    "claude": -44.89
  },
  "reward": 0.85,
  "metadata": {
    "judge_score": 0.82,
    "oracle_label": 0.90
  }
}
```

## Key Design Decisions

### 1. **Pydantic for Type Safety**
We use Pydantic models instead of plain dictionaries to:
- Catch errors early through validation
- Provide clear interfaces with IDE support
- Ensure data consistency across the pipeline

### 2. **Factory Pattern for Flexibility**
DatasetFactory separates data loading concerns from the Dataset model, allowing:
- Easy extension with new data sources
- Testability through dependency injection
- Clean separation of concerns

### 3. **Rewards as Optional**
Rewards are optional in the base Dataset but required for PrecomputedSampler because:
- Data may arrive uncalibrated (needs calibration first)
- Different estimators have different requirements
- Flexibility in pipeline design

### 4. **Metadata as Catch-All**
Non-core fields go into metadata automatically, allowing:
- Preservation of all input data
- Extension without schema changes

### 5. **Oracle Coverage Detection**
PrecomputedSampler.oracle_coverage property enables:
- Automatic oracle slice augmentation when coverage < 100%
- No configuration needed for honest confidence intervals
- Graceful handling of partial oracle labels
- Backward compatibility

### 6. **Validation at Multiple Levels**
We validate at Pydantic, structural, and semantic levels to:
- Catch issues early before expensive computation
- Provide helpful error messages
- Ensure estimation reliability

## Common Issues and Solutions

### Issue: "PrecomputedSampler requires all samples to have rewards"
**Cause**: Trying to use uncalibrated data with PrecomputedSampler
**Solution**: 
```python
from cje.calibration import calibrate_dataset

# Calibrate first
calibrated_dataset, cal_result = calibrate_dataset(
    dataset,
    judge_field="judge_score",
    oracle_field="oracle_label"
)
# Then create sampler
sampler = PrecomputedSampler(calibrated_dataset)
```

### Issue: "Log probability must be <= 0"
**Cause**: Invalid log probabilities (positive values)
**Solution**: Ensure log probs are actual log values (negative or zero)

### Issue: Missing target_policy_logprobs
**Cause**: Data doesn't have log probs for declared target policies
**Solution**: Either compute missing log probs or remove policies from target list

### Issue: Inconsistent data types in metadata
**Cause**: Mixed types in metadata fields across samples
**Solution**: Ensure consistent types or handle in preprocessing

## Performance

### Memory Considerations
- Datasets are fully loaded into memory
- For very large datasets (>1GB), consider streaming approaches
- Influence functions in EstimationResult can be large (n_samples × n_policies)
- PrecomputedSampler maintains both original and formatted data

### Optimization Tips
- `PrecomputedSampler.n_valid_samples` shows actual samples after filtering
- Invalid samples are automatically filtered during formatting
- Judge scores are accessed via `get_judge_scores()` for weight calibration

## Fold Management

The `folds` module provides unified cross-validation fold assignment across all CJE components:

### Core Functions
```python
from cje.data.folds import get_fold, get_folds_for_dataset

# Get fold for single prompt
fold = get_fold("prompt_123", n_folds=5, seed=42)  # Returns 0-4

# Get folds for entire dataset
folds = get_folds_for_dataset(dataset, n_folds=5, seed=42)

# Balanced oracle distribution (for calibration)
from cje.data.folds import get_folds_with_oracle_balance
oracle_mask = np.array([s.metadata.get("oracle_label") is not None for s in dataset.samples])
balanced_folds = get_folds_with_oracle_balance(prompt_ids, oracle_mask, n_folds=5)
```

### Key Properties
- **Deterministic**: `hash(prompt_id) % n_folds` ensures reproducibility
- **Filtering-proof**: Based on stable prompt_id, not array indices
- **Fresh-draw compatible**: Same prompt_id → same fold always
- **Cross-component consistent**: All estimators use same fold system

**Note**: Folds are computed on-demand using `hash(prompt_id) % n_folds`. The fold configuration (n_folds, fold_seed) is stored in dataset metadata for reproducibility.

## Advanced Topics

### Custom Data Sources
Implement the DataSource protocol:
```python
from typing import List, Dict, Any

class CustomDataSource:
    def load(self) -> List[Dict[str, Any]]:
        # Your loading logic
        return data
        
# Use with factory
factory = DatasetFactory()
source = CustomDataSource()
dataset = factory.loader.load_from_source(source, target_policies=["gpt4"])
```

### Fresh Draws for DR
```python
from cje.data.fresh_draws import FreshDrawDataset, FreshDrawSample

# Create fresh draws
samples = [
    FreshDrawSample(
        prompt_id="p1",
        target_policy="gpt4",
        judge_score=0.9,
        draw_idx=0
    ),
    # ... more samples
]

fresh_dataset = FreshDrawDataset(
    target_policy="gpt4",
    draws_per_prompt=5,
    samples=samples
)
```

### Custom Validation
```python
def validate_custom_requirements(data: List[Dict]) -> Tuple[bool, List[str]]:
    issues = []
    
    # Your validation logic
    for record in data:
        if "custom_field" not in record:
            issues.append("Missing custom_field")
    
    return len(issues) == 0, issues
```

## Summary

The data module provides a robust foundation for CJE analysis through type-safe models, flexible loading patterns, and comprehensive validation. It ensures data quality early in the pipeline while maintaining flexibility for different use cases and data sources.

# ============================================================================
# FILE: cje/diagnostics/README.md
# ============================================================================

# CJE Diagnostics System

## Overview

The CJE diagnostics system provides comprehensive monitoring and validation of causal inference assumptions. It follows a **push-based architecture** where estimators compute diagnostics during estimation and attach them to results.

## Core Architecture

The diagnostics system is now consolidated into a single cohesive module at `cje/diagnostics/`:

```
cje/diagnostics/
├── __init__.py          # Public API exports
├── models.py            # Data models (IPSDiagnostics, DRDiagnostics, Status)
├── weights.py           # Weight diagnostic computations (ESS, Hill, etc.)
├── overlap.py           # Overlap metrics (Hellinger affinity, auto-tuning)
├── dr.py                # DR-specific diagnostics
├── stability.py         # Stability and drift detection
├── display.py           # Display and formatting utilities
├── robust_inference.py  # Robust standard errors and inference
└── README.md           # This documentation
```

### Three-Layer Architecture

```
┌─────────────────┐
│  Data Models    │  models.py: Immutable dataclasses
│                 │  (IPSDiagnostics, DRDiagnostics)
└────────┬────────┘
         │
┌────────▼────────┐
│  Computation    │  weights.py, dr.py, stability.py:
│                 │  Pure functions for metric computation
└────────┬────────┘
         │
┌────────▼────────┐
│  Integration    │  Estimators import and use diagnostics
│                 │  during their estimate() methods
└─────────────────┘
```

### Key Design Principles

1. **Diagnostics are data, not behavior** - Dataclasses with computed properties
2. **Push-based flow** - Created during estimation, not on-demand
3. **Fail-fast with NaN** - Critical issues return NaN estimates, not exceptions
4. **Hierarchical status** - Multiple layers of safety checks
5. **Self-describing** - Objects know how to validate, summarize, and serialize themselves

## Diagnostic Classes

The system provides two main diagnostic classes that share a common interface:

### Common Interface

All diagnostic classes provide these methods:
- `validate() -> List[str]` - Self-consistency checks, returns list of issues
- `summary() -> str` - Human-readable one-line summary
- `to_dict() -> Dict` - Full serialization including enums as strings
- `to_json(indent=2) -> str` - JSON export with configurable formatting
- `to_csv_row() -> Dict` - Flat dictionary for tabular analysis

Computed properties (via `@property`):
- `filter_rate` - Fraction of samples filtered out
- `best_policy` - Policy with highest estimate
- `overall_status` - Aggregate health status
- Additional class-specific properties

### IPSDiagnostics

Base diagnostics for importance sampling estimators. Key field groups:

**Identification**: `estimator_type`, `method`, `policies`  
**Sample counts**: `n_samples_total`, `n_samples_valid`, `n_samples_used`  
**Results**: `estimates`, `standard_errors` (per policy)  
**Weight metrics**: `weight_ess`, `ess_per_policy`, `max_weight_per_policy`  
**Tail behavior**: `tail_indices` (Hill estimator results)  
**Status**: `weight_status`, `status_per_policy`  
**Calibration**: `calibration_rmse`, `calibration_r2`, `n_oracle_labels`

### DRDiagnostics  

Extends IPSDiagnostics with doubly robust specific metrics:

**Cross-fitting**: `dr_cross_fitted`, `dr_n_folds`  
**Outcome model**: `outcome_r2_range`, `outcome_rmse_mean`  
**Influence functions**: `worst_if_tail_ratio`, `influence_functions`  
**Decompositions**: `dr_diagnostics_per_policy`, `dm_ips_decompositions`  
**Orthogonality**: `orthogonality_scores`

## Status System

The diagnostic system uses a **three-tier hierarchy**:

### 1. Computed Status (Informational)
Each diagnostic object computes an `overall_status` based on its metrics. This is purely informational and shown in displays but doesn't prevent estimation.

The `Status` enum has three values:
- `GOOD` - All metrics within acceptable ranges
- `WARNING` - Some concerning metrics but results usable
- `CRITICAL` - Severe issues detected

Status computation varies by diagnostic class and combines multiple factors like ESS, tail indices, and calibration quality.

### 2. Validation Warnings  
The `validate()` method checks for logical inconsistencies:
- Impossible values (ESS > 1.0, R² > 1.0)
- Inconsistent counts (n_valid > n_total)
- Extreme metrics that suggest problems

Returns a list of issue descriptions. Empty list means all checks pass.

### 3. Refusal Gates (Hard Stops)
Some estimators refuse to provide estimates when diagnostics indicate unreliable results. When triggered, they return `NaN` rather than potentially misleading estimates.

Example: CalibratedIPS may refuse estimation based on combinations of ESS, weight concentration, and coefficient of variation. These thresholds are estimator-specific and more conservative than status levels.

## Key Diagnostic Metrics

### Hellinger Affinity (Bhattacharyya Coefficient)
Measures structural overlap between policies. **Cannot be improved by calibration.**
- **Affinity > 50%**: Good overlap
- **Affinity 35-50%**: Marginal overlap  
- **Affinity 20-35%**: Poor overlap (calibration might help)
- **Affinity < 20%**: Catastrophic mismatch (refuse estimation)

Key insight: Hellinger tells us whether to give up, ESS tells us how hard to try.

### Effective Sample Size (ESS)
Measures how many "effective" samples remain after weighting. **Can be improved by calibration.**
- **ESS > 30%**: Good overlap
- **ESS 10-30%**: Moderate overlap issues  
- **ESS < 10%**: Severe overlap problems

### Auto-Tuned ESS Thresholds
Instead of fixed thresholds, compute based on desired CI width:
```python
threshold = 0.9604 / (n * target_ci_halfwidth²)
```
For n=10,000 and ±1% target: threshold = 96%
For n=100,000 and ±1% target: threshold = 9.6%

### Hill Tail Index
Estimates tail behavior of importance weights (k = 5% of samples).
- **α ≥ 2**: Finite variance, acceptable
- **α ∈ [1, 2)**: Infinite variance, WARNING
- **α < 1**: Infinite mean, CRITICAL

### Calibration R²
Measures judge-to-oracle calibration quality.
- **R² ≥ 0.5**: Good calibration
- **R² ∈ [0, 0.5)**: Moderate calibration
- **R² < 0**: Poor calibration

### Weight Concentration
Fraction of samples with near-zero weight.
- **< 50%**: Acceptable
- **50-85%**: Concerning
- **> 85%**: Critical

## Usage Examples

### Basic Diagnostics Check
```python
from cje import analyze_dataset

results = analyze_dataset("data.jsonl", estimator="calibrated-ips")
diagnostics = results.diagnostics

# Check overall health
if diagnostics.overall_status == Status.CRITICAL:
    print("⚠️ Critical issues detected!")
    print(diagnostics.summary())
```

### Detailed Analysis
```python
# Check per-policy metrics
for policy in diagnostics.policies:
    print(f"{policy}: ESS={diagnostics.ess_per_policy[policy]:.1%}")
    if diagnostics.hellinger_per_policy:
        print(f"  Hellinger affinity={diagnostics.hellinger_per_policy[policy]:.1%}")

# For DR estimators
if isinstance(diagnostics, DRDiagnostics):
    min_r2, max_r2 = diagnostics.outcome_r2_range
    print(f"Outcome R² range: [{min_r2:.3f}, {max_r2:.3f}]")
```

### Using Overlap Metrics
```python
from cje.diagnostics.overlap import compute_overlap_metrics, diagnose_overlap_problems

# Analyze overlap for a specific policy
weights = estimator.get_raw_weights("target_policy")
metrics = compute_overlap_metrics(
    weights, 
    target_ci_halfwidth=0.01,  # Want ±1% CI
    auto_tune_threshold=True
)

# Get diagnosis and recommendations
should_proceed, explanation = diagnose_overlap_problems(metrics)
print(explanation)

# Check if calibration would help
if metrics.can_calibrate:
    print("SIMCal calibration could improve ESS")
else:
    print("Overlap too poor for calibration to help")
```

### Export for Analysis
```python
# Export to pandas for further analysis
import pandas as pd

df = pd.DataFrame(diagnostics.to_csv_row(), index=[0])
df.to_csv("diagnostics.csv")

# Or as JSON
with open("diagnostics.json", "w") as f:
    f.write(diagnostics.to_json())
```

## Diagnostic Gates

The system implements automatic gates that refuse estimation when critical issues are detected:

### CalibratedIPS Gates
The estimator refuses to provide estimates (returns NaN) when:
- ESS < 30% (less than 30% effective sample size)
- raw_near_zero > 85% (more than 85% of raw weights near zero)  
- top_5pct_weight > 30% AND cv_weights > 2.0 (high concentration with high variability)

### DR Estimator Gates
DR estimators inherit IPS gates and add warnings (but continue) when:
- Outcome model R² < 0 (indicates misspecification)
- Influence function tail ratio > 100 (heavy-tailed influence functions)

## Visualization

Weight diagnostics are displayed automatically when running `analyze_dataset.py`:
```
Weight Summary
----------------------------------------------------------------------
Policy                             ESS   Max Weight Status    
----------------------------------------------------------------------
clone                             45.2%      12.3456 GOOD      
parallel_universe_prompt          38.7%      18.9012 WARNING   
----------------------------------------------------------------------
```

Display utilities in `display.py` format diagnostics for tables and comparisons.

## Interpreting Diagnostics

### When to Trust Results

✅ **High Confidence**:
- Overall status: GOOD
- ESS > 50%
- Hill index > 2.5
- Calibration R² > 0.8
- DR: Balanced DM/IPS contributions

⚠️ **Use with Caution**:
- Overall status: WARNING
- ESS 20-50%
- Hill index 2.0-2.5
- Calibration R² 0.5-0.8
- DR: One component dominates

🔴 **Do Not Trust**:
- Overall status: CRITICAL
- ESS < 20%
- Hill index < 2.0
- Calibration R² < 0.5
- DR: Negative R² values

### Common Issues and Solutions

**Problem**: Low ESS (< 30%)
- **Cause**: Poor overlap between policies
- **Solution**: Use DR estimators with fresh draws

**Problem**: Heavy tails (Hill index < 2)
- **Cause**: Extreme importance weights
- **Solution**: Tighten variance cap in SIMCal

**Problem**: Poor calibration (R² < 0.5)
- **Cause**: Judge doesn't predict oracle well
- **Solution**: Increase oracle coverage or improve judge

**Problem**: Negative outcome model R²
- **Cause**: Model misspecification
- **Solution**: Check for distribution shift, add features

## Implementation Notes

### Memory Considerations
- Diagnostics store summary statistics, not raw data
- Influence functions stored in `EstimationResult.influence_functions`
- Can be large for many samples - consider memory when processing large datasets

### Adding New Metrics
1. Extend the dataclass in `models.py`
2. Add computation function to appropriate module
3. Call in estimator's `_build_diagnostics()`
4. Update `summary()` and `to_dict()` methods

## Advanced Topics

### Influence Function Analysis
```python
# Access influence functions (always stored)
for policy, ifs in results.influence_functions.items():
    z_scores = np.abs((ifs - np.mean(ifs)) / np.std(ifs))
    n_outliers = np.sum(z_scores > 3)
    print(f"{policy}: {n_outliers} influential points")
```

### Drift Detection
The Kendall-τ drift test is available but not integrated (Unix philosophy - you orchestrate):
```python
from cje.diagnostics import kendall_tau_drift
drift_result = kendall_tau_drift(historical_scores, current_scores)
if drift_result["tau"] < 0.5:
    print("Drift detected!")
```

## References

- **ESS**: Effective Sample Size in Importance Sampling (Kong, 1992)
- **Hill Estimator**: Hill (1975), "A Simple General Approach to Inference About the Tail of a Distribution"
- **Influence Functions**: Bickel et al. (1993), "Efficient and Adaptive Estimation"
- **TMLE Diagnostics**: van der Laan & Rose (2011), "Targeted Learning"

## Summary

The CJE diagnostics system provides:
- **Comprehensive monitoring** of all causal inference assumptions
- **Automatic safety gates** to prevent unreliable estimates
- **Clear status indicators** (GOOD/WARNING/CRITICAL)
- **Detailed metrics** for debugging issues
- **Export capabilities** for further analysis
- **Integration with visualization** for intuitive understanding

Always check diagnostics before trusting results!

# ============================================================================
# FILE: cje/estimators/README.md
# ============================================================================

# CJE Estimators

## Overview

The estimators module contains all causal inference estimation methods for unbiased off-policy evaluation of LLMs. These estimators transform logged data and importance weights into reliable policy value estimates with proper uncertainty quantification.

## Estimator Hierarchy

```
BaseCJEEstimator (abstract)
├── CalibratedIPS       # IPS with optional SIMCal calibration
├── StackedDREstimator  # Optimal stacking of DR estimators
└── DREstimator         # Doubly robust base (abstract)
    ├── DRCPOEstimator  # Basic DR with CPO
    ├── MRDREstimator   # Multiple robust DR
    └── TMLEEstimator   # Targeted maximum likelihood
```

## Core Concepts

### 1. Importance Sampling (IPS)
The foundation of off-policy evaluation. Reweights logged data to estimate performance under new policies using importance weights W = π_target/π_base.

### 2. Weight Calibration (SIMCal)
Stabilizes importance weights through monotone projection with variance control.
See `cje/calibration/README.md` for algorithm details.

### 3. Doubly Robust (DR) Estimation
Combines direct method (outcome model) with IPS correction. Provides two chances to get the estimate right - if either the outcome model OR the weights are correct, DR is consistent.

### 4. Multiple Robustness (MRDR)
Achieves robustness to:
- Outcome model misspecification (via DR)
- Propensity score misspecification (via calibrated weights)
- Both simultaneously (via cross-fitting)

### 5. Targeted Learning (TMLE)
Optimally combines outcome models and importance weights through targeted fluctuation to achieve optimal asymptotic efficiency.

### 6. Estimator Stacking
Forms an optimal convex combination of multiple DR estimators (DR-CPO, TMLE, MRDR) by minimizing the variance of the combined influence function. Uses outer split for honest inference.

## File Structure

```
estimators/
├── base_estimator.py      # Abstract base with common interface
├── calibrated_ips.py      # IPS with optional SIMCal calibration
├── stacking.py            # Optimal stacking of DR estimators
├── dr_base.py             # Doubly robust base class
├── mrdr.py                # Multiple robust DR
├── mrdr_tmle.py           # MRDR with TMLE fluctuation
├── tmle.py                # Standard TMLE
├── outcome_models.py      # Outcome model implementations
└── MRDR_OMEGA_WEIGHTS.md  # Documentation on MRDR weighting schemes
```

## Estimator Selection Guide

### Use **CalibratedIPS with calibrate=False** (raw mode) when:
- You have excellent overlap (ESS > 50%)
- You want the simplest baseline
- You don't have judge scores for calibration

### Use **CalibratedIPS** when:
- You have moderate overlap (ESS 20-50%)
- Judge scores are available
- You want variance-stabilized weights
- Fresh draws are not available
- Oracle slice augmentation is automatically enabled when partial oracle labels detected

### Use **DRCPOEstimator** when:
- You have poor overlap (ESS < 20%)
- Fresh draws are available (REQUIRED)
- You want basic doubly robust estimation

### Use **MRDREstimator** when:
- You need robustness to both weight and outcome model misspecification
- Fresh draws are available (REQUIRED for all DR methods)
- You have sufficient data for cross-fitting
- You want policy-specific outcome models

### Use **TMLEEstimator** when:
- You want optimal asymptotic efficiency
- Fresh draws are available (REQUIRED for all DR methods)
- You have well-specified models
- You need the most sophisticated estimation

### Use **StackedDREstimator** when:
- You want the best of all DR methods combined
- Fresh draws are available (REQUIRED)
- You want automatic selection of optimal weights
- You need robust performance without choosing a specific DR method
- **This is the recommended default for DR estimation**

## Refusal Gates in CalibratedIPS

CalibratedIPS includes safety mechanisms called "refusal gates" that detect when estimates would be unreliable due to poor overlap between policies. By default (`refuse_unreliable=False`), the estimator provides estimates with warnings. When enabled (`refuse_unreliable=True`), it returns NaN for unreliable policies.

### The Three Gates

1. **ESS < 30%**: Effective Sample Size below 30% means over 70% of your data is essentially ignored
2. **Raw near-zero > 85%**: More than 85% of raw importance weights are near zero (< 1e-10)
3. **Top 5% concentration > 30% AND CV > 2.0**: Top 5% of samples carry >30% weight with high variability

### Controlling Refusal Behavior

```python
# Default: Provide estimates with warnings (refuse_unreliable=False)
estimator = CalibratedIPS(sampler)  # Will warn but still estimate

# Strict mode: Return NaN for unreliable estimates
estimator = CalibratedIPS(sampler, refuse_unreliable=True)

# Via command line (analyze_dataset.py)
--estimator-config '{"refuse_unreliable": true}'  # Enable strict mode
```

### Interpreting Warnings

When refusal gates trigger warnings:
- **ESS warnings**: The estimate is dominated by a small fraction of samples
- **Raw near-zero warnings**: Severe distribution mismatch that calibration may mask
- **Concentration warnings**: A few outliers control the entire estimate

These estimates are statistically valid but practically unreliable. Consider:
1. Using policies with better overlap
2. Trying DR methods with fresh draws
3. Collecting data from more diverse base policies

## Common Interface

All estimators follow the same pattern:

```python
from cje import CalibratedIPS, PrecomputedSampler
from cje.calibration import calibrate_dataset

# 1. Calibrate dataset (if using flexible calibration)
calibrated_dataset, cal_result = calibrate_dataset(
    dataset, 
    enable_cross_fit=True,  # For DR methods
    calibration_mode='auto'  # Auto-selects monotone or two-stage
)

# 2. Create sampler with data
sampler = PrecomputedSampler(calibrated_dataset)

# 3. Initialize estimator
# DR estimators accept optional calibrator for proper index transformation
estimator = DRCPOEstimator(sampler, calibrator=cal_result.calibrator)
# or for IPS:
estimator = CalibratedIPS(sampler)

# 4. Fit and estimate
result = estimator.fit_and_estimate()

# 4. Access results
estimates = result.estimates        # Point estimates
std_errors = result.standard_errors # Standard errors
diagnostics = result.diagnostics    # Health metrics
influence = result.influence_functions  # For inference
```

## Key Design Principles

### 1. Transparency Over Silence
CalibratedIPS provides estimates with clear warnings by default when reliability is questionable, allowing users to make informed decisions. The optional strict mode (`refuse_unreliable=True`) returns `NaN` for catastrophically unreliable estimates to ensure scientific integrity.

### 2. Always Compute Influence Functions
All estimators compute and store influence functions for:
- Proper standard error estimation
- Policy comparison with covariance
- Bootstrap and robust inference
- Detecting influential observations
- CF-bits efficiency metrics (IFR, aESS)

### 3. Diagnostic Integration
Every estimator creates comprehensive diagnostics during estimation:
- IPS estimators → `IPSDiagnostics`
- DR estimators → `DRDiagnostics`
These are automatically attached to results for transparency.

### 4. Modular Design
DR estimators can leverage CalibratedIPS internally for weight computation while inheriting from BaseCJEEstimator. This ensures consistent weight handling across all estimators through composition rather than inheritance.

## Outcome Models

The `outcome_models.py` module provides regression models for DR estimation:

### IsotonicOutcomeModel
- Monotonic regression with judge scores (or calibrator's index)
- Accepts optional calibrator for two-stage calibration support
- When calibrator provided, uses calibrator.index() transformation
- No parametric assumptions
- Cross-fitting support

### LinearOutcomeModel  
- Simple linear regression baseline
- Fast and stable
- Good for debugging

### CalibratorBackedOutcomeModel
- Uses the same calibrator as rewards
- Ensures consistency between rewards and predictions
- Handles index transformation internally via predict_oof()
- Default when calibrator has standard isotonic models

### WeightedIsotonicOutcomeModel (MRDR)
- Isotonic regression with importance weighting
- Policy-specific models
- Configurable omega weights (see MRDR_OMEGA_WEIGHTS.md)

## Fresh Draws Integration

DR estimators can incorporate fresh draws (new responses from target policies). As of the latest version, DR estimators will **automatically attempt to load fresh draws** from standard locations when not explicitly provided.

### Automatic Loading
When you call `fit_and_estimate()` without adding fresh draws, DR estimators will search for them in:
1. `data/{policy}_responses.jsonl`
2. `data/responses/{policy}_responses.jsonl` 
3. `data/{policy}_fresh.jsonl`
4. `data/fresh_draws/{policy}.jsonl`

### Manual Loading
You can still explicitly add fresh draws:

```python
from cje.data.fresh_draws import FreshDrawDataset

# Add fresh draws to estimator (per policy)
fresh_data = FreshDrawDataset(samples=[...])
estimator.add_fresh_draws('target_policy', fresh_data)

# Estimate uses both logged and fresh data
result = estimator.fit_and_estimate()
```

## Refusal Gates

Estimators implement safety gates that refuse estimation when reliability thresholds are violated.

### Why Percentage-Based ESS Gates?

We use percentage-based ESS thresholds rather than absolute thresholds for several reasons:

1. **Scale Invariance**: Works consistently across datasets of any size without configuration
2. **Measures Overlap Quality**: Low ESS% indicates severe distribution mismatch, not just sample size  
3. **Practical Reliability**: When only a small fraction of data is effectively used, the estimate is dominated by a small subset
4. **System Simplicity**: One threshold concept that's self-documenting and easy to explain

Low ESS percentage means most of your data is essentially ignored - this indicates a fundamental problem with policy overlap that more samples won't fix.

### Gate Types

Each estimator implements appropriate refusal gates based on its robustness properties:

- **CalibratedIPS (raw mode)**: Very permissive gates (baseline method)
- **CalibratedIPS**: Moderate gates based on ESS and weight concentration
- **DR Estimators**: Inherit IPS gates, warn on poor outcome model fit

## Implementation Notes

### Weight Caching
Estimators cache computed weights in `_weights_cache` to avoid recomputation:
```python
self._weights_cache[policy] = calibrated_weights
```

### Influence Function Storage
Always stored in `_influence_functions` and passed to results:
```python
self._influence_functions[policy] = if_contributions
```

### Cross-Fitting
DR estimators support cross-fitting for orthogonality:
- Data split into k folds using unified `cje.data.folds` system
- Each fold gets predictions from model trained on other folds
- Prevents overfitting in outcome models
- All estimators use same deterministic fold assignments (hash(prompt_id) % k)
- Fold configuration (n_folds, fold_seed) stored in dataset metadata for reproducibility

### Variance Computation

#### Standard Variance (IPS/Calibrated-IPS)
Standard errors computed from influence functions:
```python
se = np.std(influence_functions, ddof=1) / np.sqrt(n)
```

#### Monte Carlo Variance (DR Estimators)
DR estimators now correctly include MC uncertainty from finite fresh draws:
```python
# Base variance (across-prompt)
base_var = np.var(influence_functions, ddof=1) / n

# MC variance add-on (within-prompt)
mc_var = (1/n²) × Σᵢ (1-wᵢ)² × (σ²ᵢ / Mᵢ)

# Total SE
se = np.sqrt(base_var + mc_var)
```

The MC component:
- Accounts for finite fresh draws per prompt (Mᵢ)
- Decreases as more fresh draws are collected
- Stored in `_mc_diagnostics` for transparency
- Automatically computed when fresh draws are present

## Testing

Each estimator has comprehensive tests in `cje/tests/`:
- `test_estimators.py` - Basic functionality
- `test_dr_diagnostics.py` - DR-specific tests
- `test_integration.py` - End-to-end workflows

## Advanced Topics

### Oracle Slice Augmentation (Automatic)
CalibratedIPS and DR estimators automatically detect and apply oracle slice augmentation for honest confidence intervals when partial oracle labels are available (0% < coverage < 100%). This corrects for uncertainty in the judge→oracle calibration map.

```python
# Automatic detection (default behavior)
estimator = CalibratedIPS(sampler)  # Auto-enables if oracle coverage < 100%

# Explicit control if needed
from cje.calibration import OracleSliceConfig

# Force enable
estimator = CalibratedIPS(sampler, oracle_slice_config=True)

# Force disable
estimator = CalibratedIPS(sampler, oracle_slice_config=False)

# Custom configuration
config = OracleSliceConfig(enable_augmentation=True, enable_cross_fit=True)
estimator = CalibratedIPS(sampler, oracle_slice_config=config)
```

Note: DR estimators inherit this behavior through their internal CalibratedIPS usage.

### Custom Estimators
Inherit from `BaseCJEEstimator` or `DREstimator` and implement `fit()` and `estimate()`. Always compute and store influence functions in `_influence_functions`.

### Omega Weight Configuration (MRDR)
MRDR supports different weighting schemes for outcome models:
- `"w"` (default): Most stable, uses |W|
- `"w2"`: Moderate concentration, uses W²
- `"snips"`: Extreme concentration, uses (W-1)²

See MRDR_OMEGA_WEIGHTS.md for detailed comparison.

### TMLE Fluctuation
TMLE uses iterative targeted updates with clever covariate (importance weights) to achieve optimal efficiency.

## References

- **IPS**: Horvitz & Thompson (1952)
- **Doubly Robust**: Robins et al. (1994)
- **TMLE**: van der Laan & Rubin (2006)
- **MRDR**: Multiple robustness framework (2024)

## Common Issues

- **Estimates are NaN**: Check ESS in diagnostics. Likely poor overlap - try CalibratedIPS or DR methods.
- **ESS always too low**: Policies may be too different. Consider collecting more diverse base data.
- **DR fails without fresh draws**: All DR methods REQUIRE fresh draws. Generate them first.
- **Different results between runs**: Set random seeds for reproducibility in cross-fitting.

## Summary

The estimators module provides a comprehensive toolkit for causal inference on LLM outputs, from simple importance sampling to sophisticated multiply-robust methods. Each estimator makes different bias-variance tradeoffs, but all follow the same interface and provide transparent diagnostics for reliability assessment.

# ============================================================================
# FILE: cje/experiments/arena_10k_simplified/README.md
# ============================================================================

# Arena 10k Simplified

Ablation study of CJE estimators on simulated competition data, demonstrating 13.9× ESS improvement with SIMCal.

## Quick Start

```bash
# Run comprehensive ablation studies
cd ablations/
python run_all_ablations.py

# Or run individual analysis
python analyze_dataset.py --data data/cje_dataset.jsonl --estimator calibrated-ips

# Generate plots from ablation results
cd ablations/
python analyze_results.py
```

## Files

- `ablations/` - Comprehensive ablation experiments (see ablations/README.md)
- `analyze_dataset.py` - Direct CJE analysis with detailed diagnostics  
- `experiment_config.py` - Policy definitions and experiment parameters
- `generate_arena_data.py` - Main data generation orchestrator
- `analysis/` - Modular analysis components used by analyze_dataset.py
- `data_generation/` - Scripts to reproduce dataset from scratch
  - `compute_logprobs.py` - Compute log probabilities with teacher forcing (supports multi-pass)
  - `generate_additional_passes.py` - Orchestrate multiple passes for non-determinism analysis
  - `prepare_cje_data.py` - Combine responses and logprobs into final dataset
  - `generate_responses.py` - Generate fresh responses for DR estimators
  - `add_scores_with_resume.py` - Add judge/oracle scores with resume capability
- `data/` - Main dataset with ~5000 Arena samples

## Key Results

| Method | ESS | Error vs Oracle |
|--------|-----|-----------------|
| **CalibratedIPS** | 62.7% | 0.038 |
| **RawIPS** | 4.5% | 0.175 |

**Impact**: 13.9× better ESS, 4.5× lower error, works with 2% oracle labels (20 samples)

## Data Generation Pipeline

To reproduce the dataset from scratch:

```bash
cd data_generation/

# 1. Prepare prompts and base responses
python prepare_arena_data.py

# 2. Compute log probabilities for all policies
python compute_logprobs.py --responses-dir ../data/responses --output-dir ../data/logprobs/
# This computes pass 1 (original) for all policies

# 3. Generate fresh responses for DR estimators (requires API keys)
python generate_responses.py --policy clone --n-samples 1000
# ... repeat for all policies

# 4. Add judge scores
python add_scores_with_resume.py --input ../data/responses/ --output ../data/

# 5. Create final CJE dataset
python prepare_cje_data.py --output ../data/cje_dataset.jsonl

# 6. (Optional) Generate multiple passes to study API non-determinism
source ../../../set_secrets.sh  # REQUIRED: Load API keys
python generate_additional_passes.py --data-dir ../data --n-passes 5
```

## Dataset Details

**Main Dataset**: `data/cje_dataset.jsonl` (4989 samples)
- **Policies**: 
  - `base` - Llama-70B with standard helpful assistant prompt (logging policy)
  - `clone` - Same model and prompt as base (for control/comparison)
  - `parallel_universe_prompt` - Llama-70B with parallel universe system prompt
  - `premium` - Llama-405B with standard helpful assistant prompt
  - `unhelpful` - Llama-70B with deliberately unhelpful system prompt (stress testing)
- **Scores**: 
  - Judge scores (0-1) from GPT-4.1-nano
  - Oracle labels (0-1) from GPT-5 (simulated ground truth)
- **Log probabilities**: In `data/logprobs/`
- **Fresh draws**: In `data/responses/` for DR estimators

**Note**: `unhelpful` has catastrophic overlap (ESS < 1%), returns NaN by design

## Example Commands

```bash
# Run all ablations
cd ablations/
python run_all_ablations.py

# Or run individual ablations
python oracle_coverage.py  # Test oracle coverage requirements
python sample_size.py      # Test sample size scaling
python estimator_comparison.py  # Compare all estimators
python interaction.py      # Oracle × sample interaction

# Generate plots from ablation results
python analyze_results.py

# Analyze with detailed diagnostics
python analyze_dataset.py --data "data/cje_dataset.jsonl" --estimator calibrated-ips

# Use stacked-dr for optimal DR estimation (combines DR-CPO, TMLE, MRDR)
python analyze_dataset.py --data "data/cje_dataset.jsonl" --estimator stacked-dr

# Compare estimates to oracle ground truth (shown automatically)
python analyze_dataset.py --data "data/cje_dataset.jsonl" --estimator stacked-dr --oracle-coverage 0.5
```

## Method

SIMCal calibration process:
1. Learn judge→oracle mapping via isotonic regression (2% labels)
2. Project weights onto monotone functions of judge score
3. Cap variance increase at ρ=2  
4. Result: Smooth weights, preserved unbiasedness

## Oracle Ground Truth Comparison

The analysis pipeline now includes automatic comparison of CJE estimates to oracle ground truth:
- Shows policy estimates vs actual oracle means from response files
- Displays differences and whether oracle falls within 95% CI
- Reports Mean Absolute Error and CI coverage percentage
- Available for all estimators (calibrated-ips, raw-ips, dr-cpo, mrdr, tmle, stacked-dr)

Example output:
```
📊 Oracle Ground Truth Comparison:
Policy                    Estimate     Oracle       Diff       In CI?  
clone                     0.762        0.762        -0.000     ✓       
parallel_universe_prompt  0.767        0.771        -0.004     ✓       
premium                   0.764        0.762        +0.002     ✓       
unhelpful                 0.443        0.143        +0.300     ✗       
95% CI Coverage: 75.0% (3/4)
```

Note: The `unhelpful` policy typically scores very low (< 0.2) by design and is challenging to estimate accurately due to extreme distribution shift

## Requirements

```bash
# Install CJE library
pip install -e ../../../

# Additional dependencies
pip install pandas numpy matplotlib seaborn

# For data generation (optional)
export OPENAI_API_KEY=your_key
export ANTHROPIC_API_KEY=your_key
```

## Output

Results saved to `ablations/results/` organized by experiment type:
- `oracle_coverage/` - Oracle coverage ablation results and figures
- `sample_size/` - Sample size scaling results and figures  
- `estimator_comparison/` - Estimator comparison results and figures
- `interaction/` - Interaction analysis results and figures

Each directory contains:
- `results.jsonl` - Detailed results with diagnostics
- `figure_*.png` - Visualization of results

## Known Issues

### Log Probability Issues
- **~1% null logprobs**: API returns mathematically impossible positive values for some samples
- **~18% suspicious values**: Long responses have unrealistically high log probabilities
- **Root cause**: Fireworks API bugs with teacher forcing
- **Solution**: Use multi-pass generation to identify and document non-determinism

### Multi-Pass Generation
Generate multiple passes to study API non-determinism and improve data quality:

```bash
# IMPORTANT: Must load API keys first!
source ../../../set_secrets.sh

# Generate passes 2-5 for all policies
python data_generation/generate_additional_passes.py --data-dir data --n-passes 5

# Run specific passes in parallel
python data_generation/generate_additional_passes.py \
    --data-dir data \
    --n-passes 5 \
    --parallel \
    --max-workers 4

# Analyze non-determinism (coming soon)
python data_generation/analyze_nondeterminism.py --data-dir data
```

Pass files are named: `{policy}_logprobs_pass{N}.jsonl` where N=2,3,4,5...

## Notes

- DR estimators require fresh draws in `data/responses/`
- The `unhelpful` policy intentionally has poor overlap to test refusal mechanisms
- Warnings about extra prompts in fresh draws are normal and handled correctly
- DR estimators use CJE's `load_fresh_draws_auto()` for proper fresh draw handling
- Multiple passes help identify API non-determinism (observed ~6% variance between passes)

## Citation

```bibtex
@article{cje2024,
  title={Causal Judge Evaluation: Ablation Study},
  author={CJE Team},
  year={2024},
  note={13.9× ESS improvement with 2% oracle labels}
}
```

# ============================================================================
# FILE: cje/experiments/arena_10k_simplified/ablations/README.md
# ============================================================================

# CJE Ablation Experiments

Systematic ablation studies demonstrating the value of calibrated importance sampling and doubly robust methods for off-policy evaluation.

## Quick Start

```bash
# Run all ablations (takes ~30-60 minutes)
python run_all_ablations.py

# Or run individual ablations
python oracle_coverage.py       # Effect of oracle label coverage
python sample_size.py           # Sample size scaling behavior
python estimator_comparison.py  # Compare all estimation methods
python interaction.py           # Oracle × sample size interaction

# Analyze and visualize results
python analyze_results.py

# Regenerate plots from existing data (without re-running experiments)
python regenerate_estimator_plots.py

# Or regenerate with analyze_results for other ablations
python analyze_results.py --figures
```

## What Each Ablation Tests

### 1. Oracle Coverage (`oracle_coverage.py`)
**Question**: How many oracle labels do we need for effective calibration?
- Tests: 5%, 10%, 20%, 50%, 100% oracle coverage
- Finding: 5-10% is sufficient; diminishing returns beyond 20%

### 2. Sample Size (`sample_size.py`)  
**Question**: How does performance scale with dataset size?
- Tests: n = 100, 250, 500, 1000, 2500, 5000 samples
- Compares: RawIPS, CalibratedIPS, DRCPO, CalibratedDRCPO
- Shows progression: IPS → Cal-IPS (calibration benefit) and DRCPO → Cal-DRCPO (DR + calibration)
- Finding: Cal-IPS achieves √n convergence; DR methods help at small n; calibration crucial at all scales

### 3. Estimator Comparison (`estimator_comparison.py`)
**Question**: How much does each technique improve estimates?
- Compares: 20 estimator variants across IPS and DR families
  - IPS family: RawIPS, SNIPS, CalibratedIPS
  - DR family: DRCPO, MRDR, TMLE, StackedDR (each with ±IIC, ±Calibration)
- Tests: 4×4 grid (4 sample sizes × 4 oracle coverage levels = 16 scenarios)
- Finding: Calibration provides 10-20× SE reduction; IIC adds 3-95% additional variance reduction
- Generates policy heterogeneity heatmaps showing SE and absolute error by method × policy

### 4. Interaction Effects (`interaction.py`)
**Question**: When is DR most valuable vs Cal-IPS alone?
- Tests: 3×3 grid of oracle coverage × sample size
- Finding: DR critical when n < 500 or oracle < 10%

## File Structure

```
ablations/
├── core/                       # Shared infrastructure
│   ├── base.py                # BaseAblation class for shared functionality
│   ├── schemas.py             # ExperimentSpec, result schemas
│   └── diagnostics.py         # ESS, tail index, CV metrics
├── oracle_coverage.py          # Ablation 1: Oracle coverage
├── sample_size.py             # Ablation 2: Sample size  
├── estimator_comparison.py    # Ablation 3: Method comparison
├── interaction.py             # Ablation 4: Interaction effects
├── run_all_ablations.py       # Master runner script
├── analyze_results.py         # Analysis and visualization
├── regenerate_estimator_plots.py  # Regenerate plots from existing data
└── results/                   # Generated results (auto-created)
    ├── oracle_coverage/       
    │   ├── results.jsonl      # Detailed experiment data
    │   └── figure_1_oracle_coverage.png  # Visualization
    ├── sample_size/          
    │   ├── results.jsonl
    │   └── figure_2_sample_scaling.png
    ├── estimator_comparison/ 
    │   ├── results.jsonl
    │   ├── estimator_comparison.png         # Main 16-panel comparison
    │   └── policy_heterogeneity_*.png       # Per-scenario heatmaps (32 total)
    └── interaction/          
        ├── results.jsonl
        └── figure_3_interaction.png
```

## Finding and Understanding Results

After running experiments, results are saved in the `results/` directory:

### Result Files
- **`results.jsonl`**: One JSON object per experiment containing:
  - `spec`: Experiment configuration (estimator, oracle_coverage, etc.)
  - `estimates`: Policy value estimates for each target policy
  - `standard_errors`: Uncertainty estimates
  - `diagnostics`: ESS, calibration RMSE, etc.
  - `rmse_vs_oracle`: Error compared to ground truth
  - `success`: Whether experiment completed successfully

### Visualization Options

The estimator comparison generates comprehensive visualizations:

1. **Main comparison figure** (`estimator_comparison.png`):
   - 16-panel grid (4×4) showing all scenarios
   - Each panel compares estimator performance for that scenario
   - Adaptive color scaling for readability

2. **Policy heterogeneity heatmaps** (32 total, 2 per scenario):
   - `policy_heterogeneity_n{size}_oracle{pct}pct_by_se.png` - Colored by standard error
   - `policy_heterogeneity_n{size}_oracle{pct}pct_by_abs_error.png` - Colored by absolute error
   - Shows which estimators work best for which policies
   - Includes oracle ground truth comparison when available

**Regenerating plots from existing data:**
```bash
# Quick regeneration without re-running experiments
python regenerate_estimator_plots.py

# This loads results.jsonl and creates all visualizations
# Takes ~30 seconds vs ~15 minutes for full experiments
```

### Viewing Results
```bash
# Check what results exist
ls -la results/*/

# Pretty-print a result file
python -m json.tool results/oracle_coverage/results.jsonl | head -50

# Count experiments per ablation
wc -l results/*/results.jsonl

# Generate summary visualizations
python analyze_results.py
```

## Key Results Summary

| Method | Standard Error | ESS Improvement | Notes |
|--------|---------------|-----------------|--------|
| Raw IPS | ~75× baseline | N/A | Unusable due to extreme variance |
| SNIPS | ~0.40 | 1× | Self-normalized but uncalibrated |
| Cal-IPS | ~0.02 | 13.9× | SIMCal calibration |
| DR-CPO | ~0.01-0.02 | 13.9× | Doubly robust baseline |
| DR-CPO+IIC | ~0.01 | 15-20× | IIC reduces variance 3-95% |
| Cal-DR-CPO | ~0.01-0.02 | 13.9× | With reward calibration |
| Cal-DR-CPO+IIC | ~0.01 | 15-20× | Best single DR method |
| Stacked-DR | ~0.02 | 13.9× | Optimal combination of DR methods |
| Cal-Stacked-DR+IIC | ~0.01 | 15-20× | Best overall performance |

## Implementation Details

### Data Requirements
- Uses `../data/cje_dataset.jsonl` (Arena 10k simplified)
- ~1000 samples with judge scores and oracle labels
- Fresh draws in `../data/fresh_draws/` for DR methods

### Computational Requirements
- Full suite: 30-60 minutes on standard laptop
- Individual ablations: 5-15 minutes each
- Memory: < 4GB RAM
- Storage: ~50MB for results

## Troubleshooting

**Import errors**: Run from the ablations directory:
```bash
cd cje/experiments/arena_10k_simplified/ablations
python oracle_coverage.py
```

**Missing data**: Ensure dataset exists:
```bash
ls ../data/cje_dataset.jsonl
ls ../data/fresh_draws/  # For DR methods
```

**Re-running experiments**: Simply run the script again:
```bash
python run_all_ablations.py
```

**Regenerating plots only**: If you have results but need plots:
```bash
python regenerate_estimator_plots.py  # For estimator comparison
python analyze_results.py --figures   # For other ablations
```

**Plot display issues**: Recent fixes ensure:
- All 16 scenarios are shown (was limited to 4)
- Adaptive color scaling based on data percentiles
- Automatic text color adjustment for readability
- Directory creation before saving files

## Paper Figures

Results generate figures saved to `results/`:
- `oracle_coverage_results.png` - Figure 3a in paper
- `sample_size_scaling.png` - Figure 3b in paper  
- `estimator_comparison.png` - Figure 4 in paper
- `interaction_heatmap.png` - Figure 5 in paper

# ============================================================================
# FILE: cje/experiments/arena_10k_simplified/analysis/multiple_passes/README.md
# ============================================================================

# Multiple Teacher Forcing Passes Analysis

This directory contains analysis of API non-determinism through multiple teacher forcing passes.

## Purpose

We collected 5 independent teacher forcing passes for each prompt-response pair to:
1. Study API non-determinism and its statistical properties
2. Validate the block bootstrap approach
3. Understand variance decomposition (within vs between prompts)

## Key Finding

**99.9% of variance is between prompts, only 0.1% within prompts.**

This validates treating prompts as independent blocks and shows multiple API calls provide minimal statistical benefit.

## Files

- `analyze_variance_decomposition.py` - Canonical analysis script that creates the paper figure
- `multiple_passes_findings.md` - Detailed findings and interpretation

## Usage

This analysis is primarily for validation and won't be run regularly. To reproduce:

```bash
# Ensure you have the multiple pass data in data/logprobs/*_pass*.jsonl
cd analysis/multiple_passes
python analyze_variance_decomposition.py  # Creates the paper figure
```

## Data Requirements

Requires multiple pass files in `../../data/logprobs/`:
- `{policy}_logprobs.jsonl` (pass 1)
- `{policy}_logprobs_pass{2-5}.jsonl` (additional passes)

These files are generated by `data_generation/generate_additional_passes.py`.

## Output

Visualizations are saved to `../../paper_plots/`:
- `variance_decomposition.pdf` - Publication-quality figure
- `variance_decomposition.png` - Preview version

# ============================================================================
# FILE: cje/experiments/arena_10k_simplified/validation_tests/README.md
# ============================================================================

# Validation Tests

Validation scripts confirming all CJE paper features work correctly.

## Test Files

### `test_full_pipeline.py`
Complete end-to-end validation of all features:
- IIC variance reduction (3-95%)
- SIMCal variance cap enforcement
- Fresh draw auto-loading
- Oracle slice augmentation

### `test_auto_fresh_draws.py`
Verifies automatic fresh draw loading in DR estimators.

### `test_var_cap.py`
Confirms SIMCal variance constraints are enforced.

## Running Tests

To validate all features:
```bash
cd validation_tests
python test_full_pipeline.py
```

To test specific features:
```bash
python test_auto_fresh_draws.py  # Test fresh draw auto-loading
python test_var_cap.py           # Test SIMCal variance cap
python diagnostic_check.py       # Quick diagnostic check
```

## Expected Results

All tests should show:
- ✓ IIC diagnostics found and variance reduction achieved
- ✓ SIMCal variance constraints enforced when needed
- ✓ Fresh draws auto-loaded successfully
- ✓ Oracle slice augmentation applied with partial labels

## Created During

These tests were created during the validation phase to ensure all paper claims are supported by working code (August 2024).

# ============================================================================
# FILE: cje/interface/README.md
# ============================================================================

# CJE Interface

High-level tools for using CJE. Most users should start here.

## Overview

This module provides two ways to use CJE:
1. **Python API**: `analyze_dataset()` function
2. **Command-line interface**: `python -m cje` or `cje` command

Both provide the same functionality - choose based on your workflow.

## Quick Start

### Python API

```python
from cje import analyze_dataset

# Simple analysis with defaults
results = analyze_dataset("your_data.jsonl")
print(f"Estimate: {results.estimates[0]:.3f}")
```

### Command Line

```bash
# Simple analysis with defaults
python -m cje analyze your_data.jsonl

# With options
python -m cje analyze data.jsonl --estimator dr-cpo --verbose

# Validate dataset
python -m cje validate data.jsonl --verbose
```

## Python API Reference

### `analyze_dataset()`

The main entry point for CJE analysis. Handles the complete workflow automatically.

```python
def analyze_dataset(
    dataset_path: str,
    estimator: str = "calibrated-ips",
    judge_field: str = "judge_score",
    oracle_field: str = "oracle_label",
    estimator_config: Optional[Dict[str, Any]] = None,
    fresh_draws_dir: Optional[str] = None,
    verbose: bool = False,
) -> EstimationResult:
```

**Parameters:**
- `dataset_path`: Path to JSONL file with your data
- `estimator`: Which estimator to use (see [Estimator Choice](#estimator-choice))
- `judge_field`: Field in metadata containing judge scores
- `oracle_field`: Field in metadata containing oracle labels
- `estimator_config`: Optional configuration for the estimator
- `fresh_draws_dir`: Directory with fresh draws (for DR estimators)
- `verbose`: Print progress messages

**Returns:**
- `EstimationResult` with estimates, standard errors, diagnostics, and metadata

**Example:**
```python
results = analyze_dataset(
    "data.jsonl",
    estimator="calibrated-ips",
    verbose=True
)

# Access results
for i, policy in enumerate(results.metadata["target_policies"]):
    print(f"{policy}: {results.estimates[i]:.3f} ± {results.standard_errors[i]:.3f}")

# Check diagnostics
if results.diagnostics:
    print(f"ESS: {results.diagnostics.weight_ess:.1%}")
```

## CLI Reference

### Commands

#### `analyze` - Run CJE analysis

```bash
python -m cje analyze <dataset> [options]
```

**Options:**
- `--estimator {calibrated-ips,raw-ips,stacked-dr,dr-cpo,mrdr,tmle}`: Estimation method
- `--estimator-config JSON`: Configuration as JSON string
- `--judge-field FIELD`: Judge score field name (default: judge_score)
- `--oracle-field FIELD`: Oracle label field name (default: oracle_label)
- `--verbose, -v`: Detailed output
- `--quiet, -q`: Minimal output

**Example:**
```bash
python -m cje analyze data.jsonl --estimator dr-cpo --verbose
```

#### `validate` - Check dataset format

```bash
python -m cje validate <dataset> [options]
```

**Options:**
- `--verbose, -v`: Show detailed statistics

**Example:**
```bash
python -m cje validate data.jsonl --verbose
```

## Estimator Choice

| Estimator | When to Use | Requirements |
|-----------|-------------|--------------|
| `calibrated-ips` | **Default - start here** | Judge scores |
| `raw-ips` | Diagnostic comparison | None |
| `stacked-dr` | **Best DR option** - combines all DR methods | Fresh draws |
| `dr-cpo` | Low overlap (ESS < 10%) | Fresh draws |
| `mrdr` | Research/advanced | Fresh draws |
| `tmle` | Research/advanced | Fresh draws |

## Data Format

Your JSONL file should have entries like:

```json
{
  "prompt": "What is machine learning?",
  "response": "Machine learning is...",
  "base_policy_logprob": -45.67,
  "target_policy_logprobs": {
    "policy_a": -42.31,
    "policy_b": -44.89
  },
  "metadata": {
    "judge_score": 0.82,
    "oracle_label": 0.90
  }
}
```

Required fields:
- `prompt`, `response`: The conversation
- `base_policy_logprob`: Log probability under logging policy
- `target_policy_logprobs`: Log probabilities under target policies
- `metadata.judge_score`: Score from automated judge (for calibration)

Optional:
- `metadata.oracle_label`: Ground truth labels (for calibration)
- Fresh draws in separate directory (for DR estimators)

## Workflow Examples

### Basic Analysis
```python
# Just get estimates
results = analyze_dataset("data.jsonl")
best_policy_idx = np.argmax(results.estimates)
print(f"Best policy: {results.metadata['target_policies'][best_policy_idx]}")
```

### With Diagnostics
```python
results = analyze_dataset("data.jsonl", verbose=True)

# Check quality
if results.diagnostics.weight_ess < 0.05:
    print("Warning: Very low ESS, consider DR estimator")
    results = analyze_dataset("data.jsonl", estimator="dr-cpo")
```

### Custom Configuration
```python
# Tighter variance control for CalibratedIPS
results = analyze_dataset(
    "data.jsonl",
    estimator="calibrated-ips",
    estimator_config={"var_cap": 0.5}
)

# More folds for DR
results = analyze_dataset(
    "data.jsonl", 
    estimator="dr-cpo",
    estimator_config={"n_folds": 10},
    fresh_draws_dir="fresh/"
)
```

## Interpreting Results

### Point Estimates
- `results.estimates`: Array of estimates for each target policy
- Higher is better (assuming rewards are on 0-1 scale)
- Compare policies by their estimates

### Uncertainty
- `results.standard_errors`: Standard errors for each estimate
- 95% CI: `estimate ± 1.96 * standard_error`
- Wider CIs with less oracle data (automatic adjustment)

### Diagnostics
- `weight_ess`: Effective sample size (> 0.1 is good)
- `tail_indices`: Heavy tail detection (> 2 is good)
- `calibration_r2`: Judge-oracle calibration quality (> 0.5 is good)

### CF-bits Analysis (Advanced)

For deeper reliability analysis, use CF-bits information accounting:

```python
from cje.cfbits import cfbits_report_fresh_draws, cfbits_report_logging_only

# After fitting your estimator
if has_fresh_draws:
    report = cfbits_report_fresh_draws(estimator, policy)
else:
    report = cfbits_report_logging_only(estimator, policy)

# Check reliability gates
if report["gates"]["state"] == "REFUSE":
    print("Cannot trust this estimate")
elif report["gates"]["state"] == "CRITICAL":
    print("Use with extreme caution")

# View decomposition
print(f"A-ESSF: {report['overlap']['aessf']:.1%}")  # Structural overlap
print(f"IFR: {report['efficiency']['ifr_oua']:.1%}")  # Statistical efficiency
```

CF-bits decomposes uncertainty into:
- **Identification width**: Structural limits from overlap/calibration
- **Sampling width**: Statistical noise from finite samples
- **A-ESSF**: Ceiling on S-indexed calibration methods
- **IFR**: How close to efficiency bound

## Common Issues

### "ESS too low"
- Policies are too different from logging policy
- Solution: Use DR estimator with fresh draws

### "No oracle labels found"
- Dataset missing oracle labels for calibration
- Solution: Add at least some oracle labels, or use raw rewards

### "NaN estimates"
- Catastrophic failure (usually extreme weights)
- Solution: Check data quality, use tighter var_cap

## Advanced Usage

For more control, use the library components directly:

```python
from cje import load_dataset_from_jsonl, calibrate_dataset
from cje import PrecomputedSampler, CalibratedIPS

# Manual workflow
dataset = load_dataset_from_jsonl("data.jsonl")
calibrated_dataset, _ = calibrate_dataset(dataset)
sampler = PrecomputedSampler(calibrated_dataset)
estimator = CalibratedIPS(sampler)
results = estimator.fit_and_estimate()
```

See module documentation for component details.

# ============================================================================
# FILE: cje/teacher_forcing/README.md
# ============================================================================

# CJE Teacher Forcing Module

## Overview

The teacher forcing module computes log probabilities log P(response|prompt) for importance weight calculation in CJE. It provides robust, production-ready implementations with automatic fallback mechanisms and support for various chat templates.

## When to Use

### Use **compute_teacher_forced_logprob** when:
- You need raw log P(response|prompt) for completion-style inputs
- You're working directly with the Fireworks API
- You want fine control over the computation method

### Use **compute_chat_logprob** when:
- You have chat-formatted conversations
- You need automatic template detection for Fireworks models
- You want to score assistant replies in multi-turn dialogues

### Use **Template configs** when:
- Working with specific model families (Llama, HuggingFace)
- Converting between chat and completion formats
- Ensuring correct tokenization boundaries

## File Structure

```
teacher_forcing/
├── __init__.py              # Public API exports
├── api/
│   ├── __init__.py
│   └── fireworks.py         # Fireworks API integration
├── chat.py                  # Chat conversation utilities
└── templates/               # Chat template configurations
    ├── __init__.py
    ├── base.py              # Abstract base class
    ├── fireworks.py         # Fireworks model templates
    ├── huggingface.py       # HuggingFace templates
    └── llama.py             # Llama-specific templates
```

## Core Concepts

### 1. Teacher Forcing Method
Computes log P(response|prompt) by feeding the concatenated prompt+response to the model and extracting token-level log probabilities. This avoids sampling bias from autoregressive generation.

### 2. One-Call vs Two-Call Approaches
- **One-call**: Uses byte counting to find prompt/response boundary (~89% of cases)
- **Two-call**: Fallback using difference of two API calls (100% reliability)

### 3. Chat Templates
Different models use different formatting for chat conversations. Templates handle:
- Role markers (user/assistant/system)
- Special tokens (<|begin_of_text|>, <|eot_id|>)
- Proper tokenization boundaries

## Common Interface

### Basic Teacher Forcing
```python
from cje.teacher_forcing import compute_teacher_forced_logprob

result = compute_teacher_forced_logprob(
    prompt="What is machine learning?",
    response="Machine learning is a subset of AI...",
    model="accounts/fireworks/models/llama-v3p2-3b-instruct",
    temperature=1.0
)

if result.is_valid:
    print(f"Log probability: {result.value}")
    print(f"Method used: {result.metadata['method']}")
```

### Chat Conversations
```python
from cje.teacher_forcing import compute_chat_logprob

chat = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "The answer is 4."}
]

result = compute_chat_logprob(
    chat=chat,
    model="accounts/fireworks/models/llama-v3p2-3b-instruct"
)

# Computes log P("The answer is 4." | user message + template)
```

### Custom Templates
```python
from cje.teacher_forcing import (
    HuggingFaceTemplateConfig,
    Llama3TemplateConfig,
    convert_chat_to_completions
)

# For HuggingFace models
hf_config = HuggingFaceTemplateConfig("meta-llama/Llama-3.2-3B-Instruct")

# For Llama 3 models with explicit template
llama3_config = Llama3TemplateConfig()

# Convert chat to completion format
prompt_only, prompt_plus_reply = convert_chat_to_completions(chat, hf_config)
```

## Implementation Details

### Byte Counting Algorithm
The one-call approach uses UTF-8 byte counting to find the exact boundary between prompt and response tokens:

```python
def find_boundary_by_bytes_safe(tokens, prompt, reconstructed_text):
    prompt_bytes = prompt.encode("utf-8", errors="surrogatepass")
    running = b""
    
    for idx, tok in enumerate(tokens):
        tok_bytes = tok.encode("utf-8", errors="surrogatepass")
        running += tok_bytes
        
        if len(running) == len(prompt_bytes):
            return True, idx + 1, "exact_match"
        elif len(running) > len(prompt_bytes):
            # Token spans boundary - need fallback
            return False, None, "boundary_spans_token"
```

### Two-Call Fallback
When byte counting fails (e.g., token spans boundary), the system automatically falls back to:
1. Call 1: Get log P(prompt)
2. Call 2: Get log P(prompt + response)
3. Result: log P(response|prompt) = Call 2 - Call 1

This ensures 100% reliability at the cost of an extra API call.

## Key Design Decisions

### 1. **Automatic Fallback**
Rather than failing when byte counting doesn't work, the system transparently falls back to the two-call method. This ensures reliability while optimizing for efficiency.

### 2. **Template Abstraction**
Chat templates are abstracted into configuration classes, allowing easy extension for new model families without changing core logic.

### 3. **Explicit Error Handling**
All failure modes return structured `LogProbResult` objects with clear status codes and error messages, never exceptions or magic values.

### 4. **UTF-8 Safety**
Uses `surrogatepass` error handling to deal with edge cases in tokenization, ensuring robustness with multilingual text.

### 5. **Diagnostic Metadata**
Every result includes metadata about the computation method, token counts, and failure reasons for debugging and monitoring.

## Common Issues and Solutions

### Issue: "boundary_spans_token" in metadata
**Cause**: A single token contains both prompt and response text
**Solution**: System automatically uses two-call fallback

### Issue: "echo_mismatch" error
**Cause**: API normalized whitespace or line endings differently
**Solution**: Check prompt formatting, system will use fallback

### Issue: High API latency
**Cause**: Two-call fallback doubles API requests
**Solution**: Ensure prompts don't have trailing whitespace, use shorter prompts when possible

### Issue: Template not found for model
**Cause**: Using non-Fireworks model without explicit template
**Solution**: Provide explicit `HuggingFaceTemplateConfig` or `Llama3TemplateConfig`

## Performance

### Typical Metrics
- **One-call success rate**: ~89% of requests
- **API latency**: 200-400ms (one-call), 400-800ms (two-call) 
- **Token limit**: Handles up to model's context length

### Optimization Tips
- Remove trailing whitespace from prompts
- Keep prompts under 10K characters when possible
- Reuse template configs across multiple calls
- Batch requests when computing multiple log probabilities

## Advanced Usage

### Force Two-Call Method
```python
# Skip byte counting attempt
result = compute_teacher_forced_logprob(
    prompt=prompt,
    response=response,
    model=model,
    force_two_call=True  # Always use two-call
)
```

### Custom API Configuration
```python
result = compute_teacher_forced_logprob(
    prompt=prompt,
    response=response,
    model=model,
    api_key="your-api-key",
    api_base="https://custom-endpoint.com"
)
```

### System Prompts in Chat
```python
result = compute_chat_logprob(
    chat=chat,
    model=model,
    system_prompt="You are a helpful assistant."
)
```

## Summary

The teacher forcing module provides reliable computation of log probabilities for CJE's importance weights. With automatic fallback, comprehensive template support, and production-ready error handling, it ensures accurate weight calculation across diverse models and use cases.

# ============================================================================
# FILE: cje/tests/README.md
# ============================================================================

# CJE Test Suite

## Overview

Comprehensive test suite for the Causal Judge Evaluation framework, ensuring correctness of causal inference methods, calibration algorithms, and diagnostic tools. The suite combines unit tests, integration tests, and end-to-end validation using real Arena 10K data.

**Note**: See [TEST_IMPROVEMENT_PLAN.md](../../tests/TEST_IMPROVEMENT_PLAN.md) for ongoing test suite improvements.

## When to Use

### Use **Unit Tests** when:
- Developing new calibration methods
- Adding estimator functionality
- Modifying core algorithms
- Need fast feedback during development

### Use **Integration Tests** when:
- Testing complete workflows
- Validating estimator interactions
- Checking data flow through pipelines
- Verifying cross-component behavior

### Use **Arena Sample Tests** when:
- Validating against real data
- Testing production scenarios
- Benchmarking performance
- Ensuring backward compatibility

## File Structure

```
tests/
├── conftest.py                    # Shared fixtures and utilities
├── run_all_tests.py              # Test runner script
│
├── Core Tests                    
│   ├── test_simple.py            # Minimal judge calibration
│   ├── test_pipeline.py          # End-to-end workflows
│   ├── test_integration.py       # Full system integration
│   └── test_data_models.py       # Data model validation
│
├── API Tests
│   ├── test_analysis.py          # analyze_dataset() with Arena data
│   ├── test_cli.py               # CLI commands and parsing
│   └── test_export.py            # JSON/CSV export formats
│
├── Estimator Tests
│   ├── test_dr_basic.py          # DR estimation fundamentals
│   ├── test_dr_diagnostics.py    # All estimator diagnostics
│   ├── test_stacked_simcal.py    # SIMCal weight calibration
│   ├── test_oracle_slice.py      # Oracle slice augmentation
│   └── test_custom_outcome_model.py # Custom outcome models
│
├── Diagnostic Tests
│   ├── test_new_diagnostics.py   # IPSDiagnostics, DRDiagnostics
│   ├── test_stability_diagnostics.py # Tail index, ESS, stability
│   └── test_robust_inference.py  # Inference robustness checks
│
├── Utility Tests
│   ├── test_fresh_draws.py       # Fresh draw loading
│   ├── test_teacher_forcing.py   # Chat templates, log prob
│   ├── test_validation.py        # Dataset validation
│   └── test_edge_cases.py        # Missing values, extremes
│
├── Documentation Tests
│   └── test_documentation_examples.py # README code validation
│
└── data/                          # Test datasets
    ├── arena_sample/              # Real Arena 10K subset
    └── *.jsonl                    # Synthetic test data
```

## Core Concepts

### 1. Test Categories
Tests are organized by functionality and marked with pytest markers:
- **@pytest.mark.unit** - Fast, isolated component tests
- **@pytest.mark.integration** - Multi-component workflow tests
- **@pytest.mark.slow** - Tests requiring API calls or heavy computation

### 2. Arena Sample Data
Real subset from Arena 10K evaluation:
- 100 samples with actual judge scores and oracle labels
- 4 target policies: clone, premium, parallel_universe_prompt, unhelpful
- Fresh draws for each policy enabling DR estimation
- Ground truth for validation

### 3. Fixture Architecture
Shared fixtures in `conftest.py` provide consistent test data:
- **basic_dataset**: Simple 20-sample dataset with all fields
- **dataset_with_oracle**: 50% oracle coverage for calibration testing
- **dataset_for_dr**: Cross-validation folds for DR testing
- **synthetic_fresh_draws**: Mock fresh draws for DR without files

### 4. Assertion Helpers
Standard validation functions ensure consistency:
- **assert_valid_estimation_result**: Validates EstimationResult structure
- **assert_weights_calibrated**: Checks weight calibration properties
- **assert_dataset_valid**: Comprehensive dataset validation
- **assert_diagnostics_complete**: Verifies diagnostic completeness

## Common Interface

### Running Tests

```bash
# Run all tests
poetry run pytest cje/tests/

# Run by category
poetry run pytest -m unit          # Fast unit tests only
poetry run pytest -m integration   # Integration tests
poetry run pytest -m "not slow"    # Skip slow tests

# Run specific modules
poetry run pytest cje/tests/test_analysis.py -v      # High-level API
poetry run pytest cje/tests/test_dr_diagnostics.py   # DR diagnostics
poetry run pytest cje/tests/test_simple.py::test_judge_calibration  # Single test

# With coverage
poetry run pytest --cov=cje --cov-report=html cje/tests/
```

### Writing New Tests

```python
import pytest
from cje import analyze_dataset

class TestNewFeature:
    """Test new feature functionality."""
    
    def setup_method(self):
        """Setup test data."""
        self.test_data = Path(__file__).parent / "data" / "basic_test_data.jsonl"
    
    @pytest.mark.unit
    def test_feature_basic(self, basic_dataset):
        """Test basic feature behavior."""
        result = your_feature(basic_dataset)
        assert_valid_result(result)
    
    @pytest.mark.integration
    def test_feature_with_real_data(self):
        """Test with Arena sample data."""
        result = analyze_dataset(
            "data/arena_sample/dataset.jsonl",
            your_new_parameter=True
        )
        assert result.metadata["your_feature"] == expected_value
```

## Key Design Decisions

### 1. **Real Data Testing**
Arena sample data provides ground truth validation:
- Catches regressions in production scenarios
- Validates against known-good results
- Tests all estimators with same data

### 2. **Modular Test Organization**
Tests grouped by functionality, not implementation:
- Easy to find relevant tests
- Clear what each file tests
- Parallel test execution friendly

### 3. **Shared Fixtures**
Common data patterns centralized in conftest.py:
- Consistent test data across modules
- Reduced boilerplate
- Easy to add new data patterns

### 4. **Progressive Complexity**
Tests build from simple to complex:
- `test_simple.py` - Minimal functionality
- `test_pipeline.py` - Component integration
- `test_analysis.py` - Full system with real data

## Common Issues

### "FileNotFoundError for test data"
Ensure running from project root:
```bash
cd /path/to/causal-judge-evaluation
poetry run pytest cje/tests/
```

### "Slow test execution"
Skip slow tests during development:
```bash
poetry run pytest -m "not slow" cje/tests/
```

### "Import errors"
Install package in development mode:
```bash
poetry install
# or
pip install -e .
```

## Performance

- **Unit tests**: < 1 second each
- **Integration tests**: 1-5 seconds each
- **Full suite**: ~30 seconds without slow tests
- **With slow tests**: ~2 minutes (includes API calls)

Test execution tips:
- Use `-x` to stop on first failure
- Use `-k pattern` to run tests matching pattern
- Use `--lf` to run last failed tests
- Use `-n auto` for parallel execution (requires pytest-xdist)

## Summary

The CJE test suite provides comprehensive validation through 155+ tests covering all estimators, calibration methods, and diagnostic tools. It combines fast unit tests for development, integration tests for workflow validation, and real Arena data tests for production confidence, ensuring the framework produces correct, unbiased causal estimates.

# ============================================================================
# FILE: cje/tests/README_TEST_SUITE.md
# ============================================================================

# CJE Test Suite Documentation

## Overview

The CJE test suite has been radically simplified to focus on end-to-end testing with real data. We've reduced from 28 test files (238 tests) to 5 core test files (59 tests) while maintaining comprehensive coverage of critical functionality.

## Test Structure

```
cje/tests/
├── conftest.py                 # Shared fixtures and arena data loaders
├── test_e2e_estimators.py      # E2E tests for all estimators (9 tests)
├── test_e2e_features.py        # E2E tests for features like IIC, SIMCal (8 tests)
├── test_e2e_analysis.py        # Tests for user-facing API and CLI (10 tests)
├── test_infrastructure.py      # Critical infrastructure and edge cases (14 tests)
└── test_unified_folds.py       # Fold management tests (21 tests)
```

## Running Tests

### Quick Test Commands

```bash
# Run all tests
pytest cje/tests

# Run E2E tests only
pytest cje/tests/test_e2e*.py -q

# Run specific test file
pytest cje/tests/test_e2e_estimators.py -v

# Run with specific marker
pytest cje/tests -m e2e
pytest cje/tests -m "not slow"
```

### Test Categories

- **E2E Tests** (`test_e2e_*.py`): Complete workflows using real arena data
- **Infrastructure** (`test_infrastructure.py`): Critical components like fold computation
- **Unified Folds** (`test_unified_folds.py`): Comprehensive fold management testing

## Key Features

### Real Data Testing
All E2E tests use the 100-sample arena dataset located in `cje/tests/data/arena_sample/`. This includes:
- Judge scores and oracle labels
- Response files for fresh draws
- Multiple policies (clone, premium, parallel_universe_prompt, unhelpful)

### Comprehensive Coverage
Despite having 75% fewer tests, we maintain coverage of:
- ✅ All estimators (CalibratedIPS, DR-CPO, MRDR, TMLE, Stacked-DR)
- ✅ Key features (IIC, SIMCal, oracle augmentation, cross-fitting)
- ✅ User workflows (load → calibrate → estimate → export)
- ✅ Edge cases (NaN, extreme weights, missing data)
- ✅ Infrastructure (fold computation, data validation)

### Fresh Draws
The `arena_fresh_draws` fixture properly filters fresh draws to match the dataset prompt_ids, ensuring DR estimators work correctly.

## Test Philosophy

### End-to-End Focus
Instead of testing individual functions, we test complete pipelines:
```python
# Example E2E test pattern
def test_calibrated_ips_pipeline(arena_sample):
    # 1. Calibrate dataset
    calibrated, cal_result = calibrate_dataset(arena_sample, ...)
    
    # 2. Create sampler
    sampler = PrecomputedSampler(calibrated)
    
    # 3. Run estimation
    estimator = CalibratedIPS(sampler)
    results = estimator.fit_and_estimate()
    
    # 4. Validate results
    assert len(results.estimates) == 4
    assert all(0 <= e <= 1 for e in results.estimates)
```

### Real Data Priority
- Use arena sample for integration tests
- Synthetic data only for edge cases
- Test what users actually do

## Known Issues

### Edge Case Tests
Some edge case tests in `test_infrastructure.py::TestEdgeCases` fail due to Pydantic validation requiring at least 1 sample in Dataset. These test impossible scenarios and can be ignored.

### Test Timeouts
Some tests may timeout when run in bulk. Running individual test files usually works fine.

## Adding New Tests

When adding tests, follow these guidelines:

1. **Prefer E2E tests** - Test complete workflows
2. **Use arena data** - Real data finds real bugs
3. **Keep it focused** - Each test should have one clear purpose
4. **Document intent** - Clear test names and docstrings

Example:
```python
def test_new_feature_workflow(arena_sample):
    """Test that new feature improves estimates."""
    # Use arena_sample fixture for real data
    # Test complete workflow
    # Assert on user-visible outcomes
```

## Maintenance

### Quick Health Check
```bash
# Run core E2E test to verify setup
pytest cje/tests/test_e2e_estimators.py::TestE2EEstimators::test_calibrated_ips_pipeline -v
```

### Before Commits
```bash
# Run all E2E tests
pytest cje/tests/test_e2e*.py -q

# Run infrastructure tests
pytest cje/tests/test_infrastructure.py::TestFoldInfrastructure -q
```

## Summary

The test suite has been transformed from 238 scattered unit tests to 59 focused tests that:
- Test real workflows with real data
- Catch integration issues
- Are easy to understand and maintain
- Run quickly and reliably

This represents a **73% reduction** in test count while maintaining comprehensive coverage of critical functionality.

# ============================================================================
# FILE: cje/tests/data/arena_sample/.pytest_cache/README.md
# ============================================================================

# pytest cache directory #

This directory contains data from the pytest's cache plugin,
which provides the `--lf` and `--ff` options, as well as the `cache` fixture.

**Do not** commit this to version control.

See [the docs](https://docs.pytest.org/en/stable/how-to/cache.html) for more information.


# ============================================================================
# FILE: cje/tests/data/arena_sample/responses/.pytest_cache/README.md
# ============================================================================

# pytest cache directory #

This directory contains data from the pytest's cache plugin,
which provides the `--lf` and `--ff` options, as well as the `cache` fixture.

**Do not** commit this to version control.

See [the docs](https://docs.pytest.org/en/stable/how-to/cache.html) for more information.


# ============================================================================
# FILE: cje/utils/README.md
# ============================================================================

# CJE Utils Module

## Overview

Utility functions for export and analysis in CJE. This module provides practical tools for saving estimation results and debugging extreme weight issues.

## When to Use

### Use **Export Utilities** when:
- You need to save estimation results for reporting
- You want JSON or CSV output formats
- You need to share results with non-Python tools
- You're creating reproducible analysis pipelines

### Use **Extreme Weights Analysis** when:
- Debugging weight explosion issues
- Understanding which samples dominate estimates
- Identifying problematic log probability ratios
- Generating diagnostic reports for stakeholders

## File Structure

```
utils/
├── __init__.py                  # Re-exports and backward compatibility
├── export.py                    # JSON/CSV export functions
└── extreme_weights_analysis.py # Weight debugging and reporting
```

## Core Concepts

### 1. Result Export
Converts EstimationResult objects to standard formats:
- **JSON**: Hierarchical format with metadata and diagnostics
- **CSV**: Tabular format for spreadsheet analysis
- Handles numpy arrays, NaN values, and complex nested structures

### 2. Extreme Weights Analysis
Deep dive into importance weight behavior:
- Identifies samples with highest/lowest weights
- Tracks consistently extreme samples across policies
- Computes ESS and weight statistics
- Generates both JSON and text reports


## Common Interface

### Export Results

```python
from cje.utils import export_results_json, export_results_csv

# After running estimation
result = estimator.fit_and_estimate()

# Export to JSON with full details
export_results_json(
    result,
    "results/analysis.json",
    include_diagnostics=True,
    include_metadata=True
)

# Export to CSV for Excel
export_results_csv(
    result,
    "results/summary.csv",
    include_ci=True
)
```

### Analyze Extreme Weights

```python
from cje.utils import analyze_extreme_weights

# Debug weight issues
json_report, text_report = analyze_extreme_weights(
    dataset=dataset,
    sampler=sampler,
    raw_weights_dict=raw_weights,
    calibrated_weights_dict=calibrated_weights,
    n_extreme=10,  # Top/bottom 10 samples
    output_dir=Path("diagnostics/")
)

# Reports saved to diagnostics/extreme_weights_analysis.{json,txt}
print(text_report)  # Human-readable summary
```


## Key Design Decisions

### 1. **Graceful Serialization**
Export functions handle complex types:
- Numpy arrays → lists
- NaN → null (JSON) or empty (CSV)
- Complex objects → string representations
- Never fails on serialization errors

### 2. **Comprehensive Weight Analysis**
Extreme weights analysis provides multiple views:
- Per-policy statistics
- Cross-policy patterns
- Sample-level details
- Both JSON (programmatic) and text (human) formats


## Common Issues

### "Can't serialize object to JSON"
The export functions handle most types, but custom objects may need:
```python
# Add to metadata as strings
result.metadata["custom_obj"] = str(my_custom_object)
```

### "Extreme weights report too large"
Limit number of samples analyzed:
```python
analyze_extreme_weights(..., n_extreme=5)  # Only top/bottom 5
```

## Performance

- **Export**: O(n_policies) - Fast even for large results
- **Extreme weights**: O(n_samples × n_policies) - Can be slow for large datasets

For large datasets:
- Export in batches if memory constrained
- Analyze subset of policies for extreme weights

## Summary

The utils module provides essential tools for CJE workflows: exporting results for reporting and debugging weight issues through detailed analysis. These utilities handle the practical aspects of working with CJE results in production environments.

# ============================================================================
# FILE: cje/visualization/README.md
# ============================================================================

# CJE Visualization Module

## Overview

The visualization module provides comprehensive diagnostic plots for understanding and validating CJE analysis results. It offers specialized dashboards for weight diagnostics, doubly robust diagnostics, calibration assessment, and policy estimate comparisons to help practitioners audit assumptions and interpret results.

## When to Use

### Use **Weight Dashboards** when:
- You need to diagnose weight explosion or concentration
- You want to understand effective sample size (ESS) issues
- You're comparing raw vs calibrated weight behaviors
- You need to identify which samples dominate estimates

### Use **DR Dashboard** when:
- You're using doubly robust estimators
- You need to check orthogonality assumptions
- You want to understand DM vs IPS contributions
- You need to diagnose influence function tail behavior

### Use **Calibration Plots** when:
- You want to visualize judge → oracle calibration
- You need to assess calibration quality (ECE, RMSE)
- You're comparing before/after calibration alignment
- You want to understand calibration transformations

### Use **Estimate Plots** when:
- You need to compare policy performance
- You want confidence intervals visualized
- You have oracle ground truth for validation
- You need publication-ready forest plots

## File Structure

```
visualization/
├── __init__.py              # Public API with backward-compatible aliases
├── calibration.py           # Calibration transformation and reliability plots
├── dr_dashboards.py         # Doubly robust diagnostic visualizations
├── estimates.py             # Policy performance forest plots
└── weight_dashboards.py     # Weight diagnostic dashboards (summary & detailed)
```

## Core Concepts

### 1. Weight Diagnostics
Comprehensive analysis of importance weight behavior:
- **ESS tracking**: Monitor effective sample size degradation
- **Tail analysis**: CCDF plots to identify heavy tails
- **Concentration metrics**: How many samples contribute X% of weight
- **Calibration impact**: Compare raw vs calibrated distributions
- **Judge correlation**: Optional analysis of weight-judge score relationships

### 2. DR Diagnostics
Specialized plots for doubly robust estimation:
- **Component analysis**: Direct method vs IPS correction contributions
- **Orthogonality checks**: Score function mean ± 2SE for validity
- **Influence functions**: EIF tail behavior and stability

### 3. Calibration Assessment
Visual tools for judge calibration quality:
- **Transformation curves**: Visualize f: judge → oracle mapping
- **Reliability diagrams**: Bin-wise calibration alignment
- **Improvement metrics**: ECE and RMSE before/after calibration

### 4. Estimate Visualization
Clear presentation of final results:
- **Forest plots**: Point estimates with confidence intervals
- **Policy comparison**: Visual ranking and uncertainty
- **Oracle validation**: Compare estimates to ground truth when available

## Common Interface

All visualization functions follow consistent patterns:

```python
from cje.visualization import (
    plot_weight_dashboard_summary,
    plot_weight_dashboard_detailed,
    plot_dr_dashboard,
    plot_calibration_comparison,
    plot_policy_estimates
)

# Weight diagnostics - summary dashboard (6 panels)
fig, metrics = plot_weight_dashboard_summary(
    raw_weights_dict=raw_weights,
    calibrated_weights_dict=calibrated_weights,
    save_path="diagnostics/weights_summary.png"
)

# Weight diagnostics - detailed per-policy view
fig, metrics = plot_weight_dashboard_detailed(
    raw_weights_dict=raw_weights,
    calibrated_weights_dict=calibrated_weights,
    judge_scores=judge_scores,  # Optional for correlation analysis
    save_path="diagnostics/weights_detailed.png"
)

# DR diagnostics (requires DR estimation result)
fig, summary = plot_dr_dashboard(
    estimation_result=dr_result,
    figsize=(15, 5)
)

# Calibration comparison
fig = plot_calibration_comparison(
    judge_scores=judge_scores,
    oracle_labels=oracle_labels,
    calibrated_scores=calibrated_scores,
    save_path="diagnostics/calibration.png"
)

# Policy estimates
fig = plot_policy_estimates(
    estimates={"policy_a": 0.75, "policy_b": 0.82},
    standard_errors={"policy_a": 0.02, "policy_b": 0.03},
    oracle_values={"policy_a": 0.74, "policy_b": 0.85}
)
```

## Key Design Decisions

### 1. **Multi-Panel Dashboards**
Complex diagnostics are organized into focused panels:
- Each panel answers one specific question
- Panels are visually connected but independently interpretable
- Summary metrics accompany visual diagnostics

### 2. **Dual Dashboard Approach**
Two complementary weight visualizations:
- **Summary dashboard**: 6-panel overview across all policies
- **Detailed dashboard**: Per-policy analysis with judge score correlation
- Each serves distinct analysis needs with clear naming

### 3. **Automatic Metric Computation**
Visualizations compute and display key metrics:
- ESS and effective sample percentages
- Calibration errors (ECE, RMSE)
- Weight concentration statistics
- No need for separate metric calculation

### 4. **Save Options**
All plots support optional saving:
- Automatic file extension handling
- High DPI for publication quality
- Consistent naming conventions

## Common Issues

### "No matplotlib backend"
Install matplotlib with GUI support:
```bash
pip install matplotlib[gui]
```

### "Figure too small for content"
Adjust figsize parameter:
```python
plot_weight_dashboard_summary(..., figsize=(16, 14))
```

### "Missing diagnostics object"
Ensure estimator was run with diagnostics enabled:
```python
result = estimator.fit_and_estimate(compute_diagnostics=True)
```

## Performance

- **Weight dashboards**: O(n_samples × n_policies) for metric computation
- **DR dashboards**: O(n_samples) for influence function analysis  
- **Calibration plots**: O(n_samples × n_bins) for binning operations
- **Memory**: Dashboards create temporary copies for sorting/binning

For large datasets (>100k samples), consider:
- Sampling for scatter plots
- Reducing bin counts
- Pre-computing metrics
- Using summary dashboard instead of detailed for initial analysis

## Summary

The visualization module transforms complex statistical diagnostics into interpretable visual insights. It helps practitioners validate assumptions, diagnose issues, and communicate results effectively through carefully designed multi-panel dashboards and focused diagnostic plots.

# ============================================================================
# FILE: docs/planning/README.md
# ============================================================================

# Planning Documents

This directory contains future work plans and code review notes:

- **CODEBASE_REVIEW.md** - Code quality assessment and recommendations
- **OVERLAP_METRICS_PLAN.md** - Advanced overlap metrics integration (future)
- **SIMPLIFICATION_PLAN.md** - API simplification before public release (future)

These are not current work items but rather documented plans for future improvements.

# ============================================================================
# FILE: examples/README.md
# ============================================================================

# CJE Examples

Simple, practical examples of using CJE.

## Quick Start

The simplest way to use CJE:

```python
from cje import analyze_dataset
results = analyze_dataset("your_data.jsonl")
print(f"Estimate: {results.estimates[0]:.3f}")
```

## Example Scripts

- `basic_workflows.py` - Three common CJE workflows (oracle labels, judge calibration, log prob computation)
- `oracle_slice_demo.py` - Demonstrates how confidence intervals widen with less oracle data
- `minimal_example.py` - Absolute minimal usage example

## Running Examples

```bash
# Run basic workflows
python examples/basic_workflows.py

# Run oracle slice demo  
python examples/oracle_slice_demo.py
```

## Key Lessons

1. **Start simple**: Use `analyze_dataset()` for most cases
2. **Add complexity only when needed**: Fresh draws, custom calibration, etc.
3. **Trust the defaults**: CJE's defaults are tuned for production use

# ============================================================================
# END OF DOCUMENTATION
# ============================================================================

# Summary:
# - Total README files: 21
# - Total lines of documentation: 3798
# - Modules documented: 20
