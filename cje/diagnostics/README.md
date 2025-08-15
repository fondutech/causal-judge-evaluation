# CJE Diagnostics System

## Overview

The CJE diagnostics system provides comprehensive monitoring and validation of causal inference assumptions. It follows a **push-based architecture** where estimators compute diagnostics during estimation and attach them to results.

## Core Architecture

### Three-Layer Separation

```
┌─────────────────┐
│  Data Models    │  Immutable dataclasses (IPSDiagnostics, DRDiagnostics)
│                 │  Define the shape and structure of diagnostic data
└────────┬────────┘
         │
┌────────▼────────┐
│  Computation    │  Pure functions that compute diagnostic metrics
│                 │  (ESS, Hill index, orthogonality scores, etc.)
└────────┬────────┘
         │
┌────────▼────────┐
│  Integration    │  Estimators call computation functions and
│                 │  build diagnostic objects during estimate()
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

### 1. Effective Sample Size (ESS)
Measures how many "effective" samples remain after weighting:
```
ESS = (Σw)² / Σw²
```
- **ESS > 30%**: Good overlap
- **ESS 10-30%**: Moderate overlap issues
- **ESS < 10%**: Severe overlap problems

### 2. Hill Tail Index
Estimates the tail behavior of importance weights using the Hill estimator:
```
α = k / Σ(log(X_i) - log(X_{k+1})) for top k order statistics
```
Default uses k = 5% of samples. Interpretation:
- **α ≥ 2**: Finite variance, acceptable for inference
- **α ∈ [1, 2)**: Infinite variance, WARNING status
- **α < 1**: Infinite mean, CRITICAL status
- **None**: Returned for uniform weights (undefined)

### 3. Calibration R²
Measures judge-to-oracle calibration quality:
```
R² = 1 - Var(oracle - f(judge)) / Var(oracle)
```
- **R² ≥ 0.5**: Good calibration (no impact on status)
- **R² ∈ [0, 0.5)**: Moderate calibration (WARNING status)
- **R² < 0**: Poor calibration (CRITICAL status)

### 4. Weight Concentration
Fraction of samples with near-zero weight:
```
concentration = |{w < 1e-10}| / n
```
- **< 50%**: Acceptable
- **50-85%**: Concerning
- **> 85%**: Critical concentration

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

### Detailed Weight Analysis
```python
# Get per-policy weight diagnostics
for policy in diagnostics.policies:
    ess = diagnostics.ess_per_policy[policy]
    max_w = diagnostics.max_weight_per_policy[policy]
    tail = diagnostics.tail_indices.get(policy)
    
    print(f"{policy}:")
    print(f"  ESS: {ess:.1%}")
    print(f"  Max weight: {max_w:.1f}")
    print(f"  Tail index: {tail:.2f}" if tail else "  Tail index: N/A")
```

### DR Decomposition Analysis
```python
if isinstance(diagnostics, DRDiagnostics):
    print(f"Direct method contribution: {diagnostics.dm_contribution:.1%}")
    print(f"IPS correction: {diagnostics.ips_contribution:.1%}")
    
    # Check outcome model quality
    for policy, r2 in diagnostics.dm_r2_per_policy.items():
        if r2 < 0:
            print(f"⚠️ {policy}: Negative R² ({r2:.3f}) - model misspecified")
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

Diagnostics can be visualized through the analysis module:

### Weight Diagnostics Display
When running `analyze_dataset.py`, weight diagnostics are displayed automatically:

```
Weight Summary
----------------------------------------------------------------------
Policy                             ESS   Max Weight Status    
----------------------------------------------------------------------
clone                             45.2%      12.3456 GOOD      
parallel_universe_prompt          38.7%      18.9012 WARNING   
----------------------------------------------------------------------
Overall                           41.9%      18.9012 GOOD      
```

Display utilities in `utils/diagnostics/display.py` include:
- `create_weight_summary_table()` - Formats weight diagnostics as a table
- `format_dr_diagnostic_summary()` - Formats DR diagnostics with decompositions
- `format_diagnostic_comparison()` - Compares two diagnostic objects side by side

These utilities work with both diagnostic objects and legacy dictionary formats.

### Calibration Analysis
The calibration quality is shown during data loading:
```python
# Automatically displayed when oracle_coverage is used
results = analyze_dataset(
    "data.jsonl", 
    estimator="calibrated-ips",
    oracle_coverage=0.5
)
# Shows calibration R² and RMSE
```

Note: Full visualization dashboards are described in `docs/visualization.rst`

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

## Implementation Details

### Diagnostic Computation Flow

```
Dataset → Sampler → Estimator.fit()
                           ↓
                    Estimator.estimate()
                           ↓
                 compute_weight_diagnostics()
                 compute_dr_diagnostics() [if DR]
                           ↓
                    _build_diagnostics()
                           ↓
                 Diagnostics attached to Result
                           ↓
                    User receives Result
```

### How Estimators Build Diagnostics

```python
# Pattern for IPS estimators
def _build_diagnostics(self, result):
    # 1. Compute weight metrics per policy
    for policy in policies:
        weights = self.get_weights(policy)
        w_diag = compute_weight_diagnostics(weights, policy)
    
    # 2. Determine overall status
    if ess < 0.01:
        status = Status.CRITICAL
    
    # 3. Return immutable diagnostic object
    return IPSDiagnostics(...)
```

### Memory Efficiency
- Diagnostics store summary statistics, not raw data
- Influence functions always computed and stored for ALL estimators (IPS, CalibratedIPS, DR)
- Stored in `EstimationResult.influence_functions` as Dict[str, np.ndarray]
- Essential for proper statistical inference and standard errors
- Can be large for many samples - consider memory when processing large datasets
- Per-fold diagnostics aggregated to save space
- Computed properties calculated on-demand via @property decorators

### Dependencies
- **External**: numpy, scipy.stats (minimal)
- **Internal**: Unidirectional flow from data → computation → presentation
- **No circular dependencies** or hidden state

### Extensibility
New diagnostic metrics can be added by:
1. Extending the dataclass (add field)
2. Adding computation function to `utils/diagnostics/`
3. Calling in estimator's `_build_diagnostics()`
4. Updating `summary()` and `to_dict()` methods
5. Adding visualization support if needed

## Advanced Topics

### Cross-Validation Diagnostics
For cross-fitted estimators:
```python
# Access per-fold diagnostics
for fold_idx, fold_diag in enumerate(diagnostics.fold_diagnostics):
    print(f"Fold {fold_idx}: R² = {fold_diag.dm_r2:.3f}")
```

### Influence Function Analysis
```python
# Access influence functions (always stored in EstimationResult)
if results.influence_functions:
    import numpy as np
    
    for policy, ifs in results.influence_functions.items():
        # Check for outliers
        z_scores = np.abs((ifs - np.mean(ifs)) / np.std(ifs))
        n_outliers = np.sum(z_scores > 3)
        print(f"{policy}: {n_outliers} influential points")
        
        # Influence functions are used for:
        # 1. Computing standard errors (SE = std(IF) / sqrt(n))
        # 2. Policy comparison with proper covariance
        # 3. Detecting influential observations
        # 4. Bootstrap and robust inference
```

### Custom Diagnostic Thresholds
```python
from cje.utils.diagnostics import compute_weight_diagnostics

# Use custom thresholds
diag = compute_weight_diagnostics(
    weights,
    policy_name,
    ess_critical=0.05,      # More strict
    tail_warning=3.0,        # Less strict
    concentration_limit=0.95  # More lenient
)
```

### Drift Detection (Available but Not Integrated)

The Kendall-τ drift test from the paper is **implemented** but **intentionally not integrated** into the main pipeline:

```python
from cje.utils.diagnostics import kendall_tau_drift

# User's responsibility to orchestrate
historical_scores = load_your_historical_data()
current_scores = judge.score(anchor_prompts)

drift_result = kendall_tau_drift(historical_scores, current_scores)
if drift_result["tau"] < 0.5:
    print("Severe drift detected!")
    # User decides action: refresh oracle, recalibrate, etc.
```

This follows Unix philosophy: CJE provides the computational tool, users build the monitoring workflow. Managing anchor sets and historical state is outside CJE's scope.

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