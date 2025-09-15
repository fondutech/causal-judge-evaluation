# CJE Project Documentation - All README Files
Generated on: Mon Sep 15 13:52:30 PDT 2025
=


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
├── oracle_slice.py      # Oracle slice configuration (deprecated)
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
Stacking relies on the component estimators' influence functions and does not re-fit nuisances at the stack level.

### 4. Oracle Uncertainty Quantification (Two Approaches)
When we calibrate judge scores using only a subset of oracle labels (e.g., 10% coverage), the calibration function f̂ itself has uncertainty. We handle this through two complementary mechanisms:

**Oracle Uncertainty Augmentation (OUA)**: The default approach that uses fold-jackknife to add a **variance** component to CIs, accounting for calibration-induced uncertainty. Used by all Cal-IPS/DR estimators.

**Oracle Slice Augmentation**: An optional point-estimate **bias correction** term `(L/π_L)m̂(S)(Y-f̂(S))` used **only** in TR-CPO under MAR with fitted π_L(S), or optionally as an MCAR engineering fallback (off by default).

### 5. Isotonic Influence Control (IIC)
A variance reduction technique that residualizes influence functions against judge scores. By fitting E[φ|S] using spline or isotonic regression and computing residuals φ̃ = φ - Ê[φ|S], IIC reduces variance without changing the estimand. This is "free" variance reduction that's enabled by default in estimators that support it (CalibratedIPS, OrthogonalizedIPS, DR-CPO, and all DR variants).

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
- **New**: Supports fit/predict separation for honest inference
  - `fit()`: Learn isotonic models and mixture weights on training data
  - `predict()`: Apply learned calibration to new data with score clipping
  - `fit_transform()`: Backward-compatible single-pass method

### `oracle_slice.py` - Oracle Slice Augmentation
Implements the optional point-estimate bias correction (used primarily in TR-CPO):
- **Problem**: We use f̂(S) everywhere but only have true Y on oracle subset  
- **Solution**: Add augmentation term `(L/π_L) * m̂(S) * (Y - f̂(S))` where:
  - L indicates oracle label presence, π_L = labeling propensity
  - m̂(S) = E[W|S] estimated via isotonic regression
  - Unbiased correction for proxy-truth gap under MAR/MCAR
- **Usage**: Enabled in TR-CPO for MAR setting; optional MCAR fallback (off by default)
- **Note**: This is separate from OUA jackknife variance (the default uncertainty method)

### `iic.py` - Isotonic Influence Control
Advanced variance reduction through influence function residualization:

**Core Mechanism:**
- `IsotonicInfluenceControl`: Main class that residualizes influence functions against judge scores
- Fits E[φ|S] using either spline regression (default) or isotonic regression
- Returns residuals φ̃ = φ - Ê[φ|S] with guaranteed variance reduction
- **Critical**: Centers fitted values to preserve mean exactly (E[φ̃] = E[φ] = 0)

**Implementation Features:**
- **Flexible regression modes**:
  - Spline regression (default): Cubic splines with configurable knots for smooth fits
  - Isotonic regression: Monotone fit with automatic direction selection via Spearman correlation
- **Cross-fitting support**: Uses same folds as reward calibration for consistency
- **Automatic fallback**: Handles edge cases (insufficient data, non-finite values) gracefully
- **Comprehensive diagnostics**: R², variance reduction ratio, ESS improvement, regression type used

**Configuration via `IICConfig`:**
- `use_splines`: Enable spline regression (default=True, more flexible than isotonic)
- `n_knots`: Number of spline knots (default=8)
- `spline_degree`: Degree of spline polynomials (default=3 for cubic)
- `use_cross_fit`: Apply fold-honest fitting (default=True)
- `min_samples_for_iic`: Minimum samples required (default=50)

**Key Properties:**
- **Variance-only**: Point estimates remain unchanged, only standard errors are reduced
- **Guaranteed improvement**: Var(φ̃) ≤ Var(φ) by construction
- **Typical reductions**: 5-20% SE reduction depending on R²(φ|S)
- **Free lunch**: No additional data or assumptions required
- **Enabled by default**: All estimators use IIC unless explicitly disabled

**Why it works**: Influence functions often correlate with judge scores because both relate to outcome quality. By removing the predictable component E[φ|S], we eliminate systematic variation while preserving the estimand.

## Key Design Decisions

### 1. **Separation of Concerns**
Each calibration type is isolated with clear interfaces:
- Reward calibration doesn't know about weights
- Weight calibration doesn't know about rewards
- Outcome models are separate from both

### 2. **Mean Preservation**
Calibrations preserve means for unbiased estimation:
- Isotonic preserves the **slice sample mean** exactly, and the **population mean asymptotically** under J₁ (representative slice)
- Weight projections preserve the **sample** mean-one exactly (Hájek normalization)
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

### Isotonic Influence Control (IIC)
```python
from cje.calibration import IsotonicInfluenceControl, IICConfig

# Configure IIC with spline regression
config = IICConfig(
    use_splines=True,      # Use flexible splines instead of isotonic
    n_knots=8,            # Number of spline knots
    spline_degree=3,      # Cubic splines
    use_cross_fit=True,   # Fold-honest fitting
    compute_diagnostics=True
)

# Apply IIC to reduce influence function variance
iic = IsotonicInfluenceControl(config)
residualized_if, diagnostics = iic.residualize(
    influence=influence_function,
    judge_scores=judge_scores,
    policy="target_policy",
    fold_ids=fold_ids  # Optional: for cross-fitting
)

print(f"R²(φ|S): {diagnostics['r_squared']:.3f}")
print(f"Variance reduction: {diagnostics['var_reduction']:.1%}")
print(f"Regression type: {diagnostics['regression_type']}")

# IIC is automatically applied in estimators when use_iic=True (default)
from cje import CalibratedIPS
estimator = CalibratedIPS(sampler, use_iic=True)  # Default
```

### Oracle Uncertainty (Default: OUA Jackknife)
```python
from cje import CalibratedIPS

# Default: OUA jackknife for oracle uncertainty (recommended)
estimator = CalibratedIPS(sampler, oua_jackknife=True)  # Default
result = estimator.fit_and_estimate()
# Result has both standard_errors and robust_standard_errors

# Optional: Enable bias correction augmentation (engineering fallback)
from cje.calibration import OracleSliceConfig
oracle_config = OracleSliceConfig(
    enable_augmentation=True,
    enable_cross_fit=True,
    min_pi=0.01,
    use_mar=False  # MCAR assumption
)

estimator = CalibratedIPS(
    sampler,
    oracle_slice_config=oracle_config
)

# The augmentation automatically adjusts standard errors
# to account for calibration uncertainty
result = estimator.fit_and_estimate()

# Check oracle uncertainty via OUA jackknife (if enabled)
if result.robust_standard_errors is not None:
    print(f"Standard SE: {result.standard_errors[0]:.4f}")
    print(f"OUA-adjusted SE: {result.robust_standard_errors[0]:.4f}")
    oracle_var = result.robust_standard_errors[0]**2 - result.standard_errors[0]**2
    print(f"Oracle uncertainty contribution: {oracle_var:.6f}")
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
- **Stacked SIMCal**: O(nK²) time, O(K²) memory (K=3 candidates)
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

# Check oracle coverage (affects OUA jackknife when < 1.0)
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
- Automatic OUA jackknife activation when coverage < 100%
- Honest confidence intervals via robust_standard_errors
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

### 3. Refusal Gates (Optional)
Estimators can optionally refuse to provide estimates when diagnostics indicate unreliable results. By default, estimators **warn** and still provide estimates. When `refuse_unreliable=True`, they return `NaN` for unreliable policies.

Gate criteria use combinations of ESS, weight concentration, and coefficient of variation. These thresholds are more conservative than status levels and are estimator-specific.

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
Instead of fixed thresholds, compute based on desired CI width using variance bounds for bounded rewards [0,1]:
```python
# For bounded rewards: Var(V_IPS) ≤ 1/(4n·ESS_fraction)  
# 95% CI halfwidth: ≈ 1.96/(2√(n·ESS_fraction))
# Solving: ESS_fraction ≥ (1.96/2)²/(n·target²) = 0.9604/(n·target²)
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
├── CalibratedIPS              # IPS with optional SIMCal calibration
│   └── OrthogonalizedCalibratedIPS  # OC-IPS with robustness to f̂ and m̂ errors
├── StackedDREstimator         # Optimal stacking of DR estimators
└── DREstimator                # Doubly robust base (abstract)
    ├── DRCPOEstimator         # Basic DR with CPO
    ├── OrthogonalizedCalibratedDRCPO  # OC-DR-CPO with first-order insensitivity
    ├── MRDREstimator          # Multiple robust DR
    ├── TMLEEstimator          # Targeted maximum likelihood
    └── TRCPOEstimator         # Triply robust CPO
```

## Core Concepts

### 1. Importance Sampling (IPS)
The foundation of off-policy evaluation. Reweights logged data to estimate performance under new policies using importance weights W = π_target/π_base.

### 2. Weight Calibration (SIMCal)
Stabilizes importance weights through monotone projection with variance control.
**Important**: Weight calibration is independent of reward calibration - DR estimators always use calibrated rewards when oracle coverage < 100%, but weight calibration is optional.

**Outer CV for Honest Inference**: CalibratedIPS now supports outer cross-validation (`use_outer_cv=True`) to account for weight learning uncertainty. This provides honest standard errors by learning weights on V-1 folds and applying to the held-out fold.
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
Forms an optimal convex combination of multiple DR estimators (DR-CPO, TMLE, MRDR, OC-DR-CPO, TR-CPO-E) by minimizing the variance of the combined influence function using regularized covariance matrices. The simplified implementation uses the oracle influence curve approach (w₀ᵀφ(Z)) for proper inference, as weight learning is O_p(n^{-1}) and doesn't affect asymptotic distribution.

**Key Features:**
- **Oracle IC approach**: Direct computation of w^T φ(Z) with theoretically justified simplifications
- **OUA support**: Stacked oracle-augmented jackknife via linear combination of component jackknife paths
- **IF hygiene**: Per-component IFs are aligned on common indices, sign-aligned, and centered
- **Numerical stability**: Ridge regularization with condition number monitoring and diagnostics
- **Weight shrinkage**: Optional shrinkage toward uniform weights for stability (default 5%)
- **MC variance**: Aggregated conservatively as Σ α_k²·mc_var_k when components use fresh draws

**Implementation (~700 lines)**: The clean implementation in `stacking.py` removes ~1200 lines of unnecessary complexity (no honest CV, no outer splits, no two-way clustering) while maintaining theoretical validity through the oracle IC approach.

### 7. Orthogonalized Estimators
Achieve first-order insensitivity to nuisance estimation errors through cross-fitting:
- **OC-IPS**: Robust to errors in reward calibration f̂(S) and weight calibration m̂(S)
- **OC-DR-CPO**: Additionally robust to outcome model errors q̂(X,A), providing first-order insensitivity to f̂, m̂, q̂ errors

### 8. Triply Robust Estimation (TR-CPO)
Achieves robustness to misspecification in three components simultaneously:
- Weight calibration errors (via raw/Hájek weights)
- Reward calibration errors (via label propensity correction)
- Outcome model errors (via DR formulation)

Two variants available:
- **TR-CPO**: Uses raw weights W in correction term (theoretical form, high variance)
- **TR-CPO-E**: Uses m̂(S)=E[W|S] in correction term (efficient, variance-reduced, recommended)

The correction term is: (L/π̂_L) × [weights] × (Y-R), where weights is either W or m̂(S).
Uses cross-fitted label propensity π̂_L to correct for oracle label selection bias under MAR assumptions.

## File Structure

```
estimators/
├── base_estimator.py      # Abstract base with common interface
├── calibrated_ips.py      # IPS with optional SIMCal calibration
├── orthogonalized_ips.py  # OC-IPS with robustness to calibration errors
├── stacking.py            # Optimal stacking of DR estimators
├── dr_base.py             # Doubly robust base class
├── orthogonalized_calibrated_dr.py  # OC-DR-CPO with first-order insensitivity
├── mrdr.py                # Multiple robust DR
├── mrdr_tmle.py           # MRDR with TMLE fluctuation
├── tmle.py                # Standard TMLE
├── tr_cpo.py              # Triply robust CPO
├── outcome_models.py      # Outcome model implementations
└── MRDR_OMEGA_WEIGHTS.md  # Documentation on MRDR weighting schemes
```

## Default Recommendation

**Use StackedDREstimator** - This is the recommended default for all estimation tasks. It automatically combines multiple DR methods (DR-CPO, TMLE, MRDR, OC-DR-CPO, TR-CPO‑E) via optimal weighting to minimize variance. The implementation includes covariance regularization to ensure numerical stability when component estimators are highly correlated. Requires fresh draws.

For specific requirements or debugging, individual estimators are available but StackedDR typically provides modest improvements (1-5% SE reduction) over the best single method when components are correlated.

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
# DR estimators accept optional reward_calibrator for proper index transformation
estimator = DRCPOEstimator(sampler, reward_calibrator=cal_result.calibrator)
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

# MC variance add-on (within-prompt) for DM term
mc_var = (1/n²) × Σᵢ (σ²ᵢ / Mᵢ)

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

### Stacked DR Implementation Details

The StackedDREstimator uses regularized covariance estimation to handle highly correlated component estimators:

```python
# Key parameters
estimator = StackedDREstimator(
    sampler,
    estimators=['dr-cpo', 'tmle', 'mrdr', 'oc-dr-cpo', 'tr-cpo-e'],  # Default components
    covariance_regularization=1e-4,                                   # Regularization strength
    n_folds=20,                                                       # Cross-fitting folds
    parallel=False                                                    # Sequential processing
)
```

**Key Design Decisions:**
1. **Regularized Covariance**: Adds λI to covariance matrix (default λ=1e-4) to handle near-singular matrices when components are highly correlated
2. **Oracle IC Approach**: Uses w₀ᵀφ(Z) where w₀ are the population-optimal weights, valid because weight learning is O_p(n^{-1})
3. **Unified Fresh Draws**: All components share the same fresh draws to ensure consistent IF computation

Other DR‑family estimators (e.g., OC‑DR‑CPO, TR‑CPO) can be included, but the default keeps a compact, highly correlated trio.

### Inference for Stacked DR

- Default inference uses the oracle‑IC path: weights are learned once on the aligned IF matrix; SE is computed from the stacked IF (sd/√n). When fold IDs are available, a cluster‑robust variant can be used on the stacked IF to account for within‑fold dependence. Calibrator OUA remains at the component level and is not re‑estimated at the stack level.

**Diagnostics Provided:**
- Condition numbers (pre/post regularization)
- Component weights for each policy
- Eigenvalues of covariance matrix
- Valid vs failed estimators

### Oracle Uncertainty Augmentation (OUA Jackknife)
All estimators support optional Oracle Uncertainty Augmentation via delete-one-fold jackknife recomputation. This accounts for finite-sample uncertainty in the learned calibrator f̂(S) by providing oracle-uncertainty-adjusted standard errors in `robust_standard_errors`.

```python
# Enable OUA jackknife (requires cross-fitted calibrator)
estimator = CalibratedIPS(sampler, oua_jackknife=True)  # Default: True

# Disable OUA jackknife  
estimator = CalibratedIPS(sampler, oua_jackknife=False)

# Access OUA-adjusted standard errors
result = estimator.fit_and_estimate()
standard_ses = result.standard_errors        # Standard influence function SEs
robust_ses = result.robust_standard_errors   # Standard + oracle uncertainty
```

**Key Properties:**
- Does NOT modify point estimates (only widens confidence intervals)
- Requires cross-fitted calibrator (enable_cross_fit=True)
- Delete-one-fold recomputation: SE_robust = √(SE_main² + Var_oracle)
- Available for all estimators (IPS, DR, MRDR, TMLE, TR-CPO)
- Provides honest inference accounting for calibrator uncertainty

Note: StackedDR uses component‑level OUA (from the underlying DR/TMLE/MRDR estimators); OUA is not re‑run at the stack level.

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

## Advanced Features

### Honest Inference with Outer CV (Default Enabled)
CalibratedIPS now uses outer cross-validation by default for honest standard errors:

```python
# Outer CV is now enabled by default
estimator = CalibratedIPS(sampler)  # use_outer_cv=True by default

# To disable outer CV (not recommended):
estimator = CalibratedIPS(
    sampler,
    use_outer_cv=False      # Reverts to single-pass calibration
)

# To customize outer CV settings:
estimator = CalibratedIPS(
    sampler,
    n_outer_folds=10,       # Increase folds for more stable estimates
    honest_iic=True         # Enable honest IIC for additional variance reduction
)
```

This addresses systematic underestimation of standard errors by accounting for weight learning uncertainty.

### Honest IIC (Experimental)
When using outer CV, honest isotonic influence control can be enabled:

```python
estimator = CalibratedIPS(
    sampler,
    use_outer_cv=True,
    honest_iic=True         # Fit IIC on train folds, apply to test
)
```

This provides additional variance reduction while maintaining honesty. The IIC model is learned on training folds and applied to test folds, with automatic R² gating (skips if R² < 0.02).

## Common Issues

- **Estimates are NaN**: Check ESS in diagnostics. Likely poor overlap - try CalibratedIPS or DR methods.
- **ESS always too low**: Policies may be too different. Consider collecting more diverse base data.
- **DR fails without fresh draws**: All DR methods REQUIRE fresh draws. Generate them first.
- **Different results between runs**: Set random seeds for reproducibility in cross-fitting.
- **Underestimated SEs**: Enable `use_outer_cv=True` for honest inference that accounts for weight learning.

## Summary

The estimators module provides a comprehensive toolkit for causal inference on LLM outputs, from simple importance sampling to sophisticated multiply-robust methods. Each estimator makes different bias-variance tradeoffs, but all follow the same interface and provide transparent diagnostics for reliability assessment.


# ============================================================================
# FILE: cje/tests/README.md
# ============================================================================

# CJE Test Suite

## Overview

The CJE test suite has been radically simplified to focus on end-to-end testing with real data. We've reduced from 28 test files (238 tests) to 7 core test files (~80 tests) while maintaining comprehensive coverage of critical functionality.

## File Structure

```
tests/
├── conftest.py                    # Shared fixtures and arena data loaders
├── run_all_tests.py              # Test runner script
│
├── E2E Tests                    
│   ├── test_e2e_estimators.py    # Complete pipelines for all estimators
│   ├── test_e2e_features.py      # IIC, SIMCal, cross-fitting
│   └── test_interface_integration.py # High-level API testing
│
├── Core Tests
│   ├── test_infrastructure.py    # Critical infrastructure and edge cases
│   ├── test_unified_folds.py     # Comprehensive fold management
│   ├── test_mc_variance.py       # Monte Carlo variance testing
│   └── test_cfbits.py           # CF-bits framework tests
│
└── data/                          # Test datasets
    ├── arena_sample/              # Real Arena 10K subset (100 samples)
    │   ├── dataset.jsonl          # Main dataset with judge scores
    │   └── responses/             # Fresh draws for DR estimation
    └── *.jsonl                    # Synthetic test data for edge cases
```

## Core Concepts

### 1. End-to-End Focus
Instead of testing individual functions, we test complete pipelines:
- Load data → Calibrate → Create sampler → Estimate → Validate results
- All E2E tests use real Arena data for authentic testing
- Tests verify user-visible outcomes, not implementation details

### 2. Arena Sample Data
Real subset from Arena 10K evaluation:
- 100 samples with actual judge scores and oracle labels
- 4 target policies: clone, premium, parallel_universe_prompt, unhelpful
- Fresh draws for each policy enabling DR estimation
- Ground truth for validation

### 3. Fixture Architecture
Shared fixtures in `conftest.py` provide consistent test data:
- **arena_sample**: Real 100-sample Arena dataset
- **arena_fresh_draws**: Filtered fresh draws matching dataset prompts
- **arena_calibrated**: Pre-calibrated Arena dataset
- **synthetic datasets**: Edge case testing (NaN, extreme weights)

### 4. Test Philosophy
- **Real Data Priority**: Use arena sample for integration tests
- **Complete Workflows**: Test what users actually do
- **Fast Feedback**: Most tests run in < 1 second
- **Clear Intent**: Each test has one clear purpose

## Running Tests

```bash
# Run all tests
poetry run pytest cje/tests/

# Run E2E tests only (recommended for quick validation)
poetry run pytest cje/tests/test_e2e*.py -q

# Run specific test files
poetry run pytest cje/tests/test_e2e_estimators.py -v
poetry run pytest cje/tests/test_unified_folds.py

# Run with markers
poetry run pytest cje/tests -m e2e
poetry run pytest cje/tests -m "not slow"

# With coverage
poetry run pytest --cov=cje --cov-report=html cje/tests/

# Quick health check (single E2E test)
poetry run pytest cje/tests/test_e2e_estimators.py::TestE2EEstimators::test_calibrated_ips_pipeline -v
```

## Writing New Tests

When adding tests, follow these guidelines:

1. **Prefer E2E tests** - Test complete workflows
2. **Use arena data** - Real data finds real bugs
3. **Keep it focused** - Each test should have one clear purpose
4. **Document intent** - Clear test names and docstrings

```python
def test_new_feature_workflow(arena_sample):
    """Test that new feature improves estimates."""
    # 1. Calibrate dataset
    calibrated, cal_result = calibrate_dataset(
        arena_sample,
        judge_field="judge_score",
        oracle_field="oracle_label"
    )
    
    # 2. Create sampler
    sampler = PrecomputedSampler(calibrated)
    
    # 3. Run estimation with new feature
    estimator = YourEstimator(sampler, new_feature=True)
    results = estimator.fit_and_estimate()
    
    # 4. Validate results
    assert len(results.estimates) == 4  # 4 policies
    assert all(0 <= e <= 1 for e in results.estimates)
    # Test that new feature had expected effect
    assert results.metadata["new_feature_applied"] == True
```

## Key Design Decisions

### 1. **Simplified Test Suite**
Reduced from 238 tests to ~80 focused tests:
- 73% reduction in test count
- Comprehensive coverage maintained
- Faster execution and easier maintenance
- Focus on integration over unit testing

### 2. **Real Data Testing**
Arena sample data provides ground truth validation:
- Catches regressions in production scenarios
- Tests all estimators with same data
- Reveals integration issues unit tests miss

### 3. **E2E Testing Priority**
Complete workflows over isolated functions:
- Test what users actually do
- Catch integration bugs
- Validate full pipelines
- Ensure components work together

### 4. **Unified Fold System**
Consistent cross-validation across all components:
- Hash-based fold assignment from prompt_id
- Prevents data leakage
- Ensures reproducibility
- Single source of truth (`data/folds.py`)

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

- **E2E tests**: < 2 seconds each
- **Infrastructure tests**: < 1 second each  
- **Full suite**: ~15 seconds for all tests
- **CF-bits tests**: May be slower due to complex computations

Test execution tips:
- Use `-x` to stop on first failure
- Use `-k pattern` to run tests matching pattern
- Use `--lf` to run last failed tests
- Use `-q` for quiet output during development
- Run E2E tests first for quick validation

## Summary

The CJE test suite has been transformed from 238 scattered unit tests to ~80 focused tests that test real workflows with real data. This simplified approach catches more integration issues, runs faster, and is easier to maintain while providing comprehensive coverage of all estimators, calibration methods, and diagnostic tools.

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
# END OF DOCUMENTATION
# ============================================================================

# Summary:
# - Total README files: 7
# - Total lines of documentation: 4334
# - Modules documented: 15
