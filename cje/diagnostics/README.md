# CJE Diagnostics System

## Overview

The CJE diagnostics system provides comprehensive monitoring and validation of causal inference assumptions. Every estimator produces structured diagnostics that expose potential issues with overlap, calibration, and model fit.

## Core Philosophy

Following CLAUDE.md principles:
- **Explicit over implicit**: All diagnostic thresholds are configurable
- **Fail fast and clearly**: Critical issues trigger immediate warnings
- **Do one thing well**: Each diagnostic class has a single purpose
- **No hidden state**: Diagnostics are immutable data structures

## Diagnostic Classes

### 1. IPSDiagnostics
For importance sampling estimators (RawIPS, CalibratedIPS).

```python
@dataclass
class IPSDiagnostics:
    # Core metrics
    estimator_type: str            # "RawIPS" or "CalibratedIPS"
    n_samples_valid: int           # Samples used after filtering
    weight_ess: float              # Effective sample size fraction
    weight_status: Status          # GOOD, WARNING, or CRITICAL
    
    # Per-policy metrics
    ess_per_policy: Dict[str, float]
    max_weight_per_policy: Dict[str, float]
    tail_indices: Dict[str, float]  # Hill tail index (α < 2 is bad)
    
    # Calibration (if applicable)
    calibration_rmse: Optional[float]
    calibration_r2: Optional[float]
```

### 2. DRDiagnostics
For doubly robust estimators (DR-CPO, MRDR, TMLE).

```python
@dataclass
class DRDiagnostics:
    # Inherits all IPSDiagnostics fields plus:
    
    # Outcome model quality
    dm_rmse_per_policy: Dict[str, float]    # Direct method RMSE
    dm_r2_per_policy: Dict[str, float]      # Outcome model R²
    
    # Decomposition
    dm_contribution: float                   # % from direct method
    ips_contribution: float                  # % from importance sampling
    
    # Influence functions
    worst_if_tail_ratio: float              # Tail heaviness of EIF
    if_normality_pvalue: float              # Shapiro-Wilk test
    
    # TMLE-specific
    tmle_convergence_iter: Optional[int]
    tmle_max_score_z: Optional[float]       # Orthogonality check
```

## Status Levels

### GOOD ✅
- ESS > 30%
- Hill tail index > 2.5
- Calibration R² > 0.7
- All checks pass

### WARNING ⚠️
- ESS 10-30%
- Hill tail index 1.5-2.5
- Calibration R² 0.3-0.7
- Some concerns but results usable

### CRITICAL 🔴
- ESS < 10%
- Hill tail index < 1.5 (infinite mean)
- Calibration R² < 0.3
- Results likely unreliable

### UNKNOWN ❓
- Insufficient data for diagnostics
- Computation failed
- Not applicable for estimator

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
Estimates the tail behavior of importance weights:
```
α = 1 / mean(log(w[i]/w[k+1])) for largest k weights
```
- **α > 2.5**: Light tails, stable
- **α ∈ [2, 2.5]**: Moderate tails
- **α ∈ [1.5, 2]**: Heavy tails, infinite variance
- **α < 1.5**: Extremely heavy, infinite mean

### 3. Calibration R²
Measures judge-to-oracle calibration quality:
```
R² = 1 - Var(oracle - f(judge)) / Var(oracle)
```
- **R² > 0.7**: Strong calibration
- **R² ∈ [0.3, 0.7]**: Moderate calibration
- **R² < 0.3**: Weak calibration

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
```python
# Refuses estimation if:
- ESS < 30% AND raw_near_zero > 85%
- ESS < 10% (always)
- Hill index < 1.0 (infinite mean)
```

### DR Estimator Gates
```python
# Warns but continues if:
- Outcome model R² < 0
- Influence function tail ratio > 100
- TMLE orthogonality |z| > 2
```

## Visualization

Diagnostics integrate with the visualization system:

### Weight Dashboard
```python
from cje.visualization import create_weight_dashboard

# Generates comprehensive weight analysis plot
create_weight_dashboard(
    estimator,
    sampler,
    dataset,
    output_path="weight_diagnostics.png"
)
```

Shows:
- Weight distributions per policy
- ESS across policies
- Concentration patterns
- Tail behavior

### Calibration Plots
```python
from cje.visualization import plot_calibration_analysis

# Shows judge-oracle calibration quality
plot_calibration_analysis(
    dataset,
    calibration_result,
    output_path="calibration.png"
)
```

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
1. **During fit()**: Collect weight statistics, fit outcome models
2. **During estimate()**: Compute influence functions, decompositions
3. **Post-estimation**: Aggregate into diagnostic objects
4. **Validation**: Run self-consistency checks

### Memory Efficiency
- Diagnostics store summary statistics, not raw data
- Influence functions optionally stored (`store_influence=True`)
- Per-fold diagnostics aggregated to save space

### Extensibility
New diagnostic metrics can be added by:
1. Extending the dataclass
2. Adding computation in estimator
3. Updating `summary()` and `to_dict()` methods
4. Adding visualization support

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
# Get influence functions (if stored)
if "dr_influence" in results.metadata:
    import numpy as np
    
    for policy, ifs in results.metadata["dr_influence"].items():
        # Check for outliers
        z_scores = np.abs((ifs - np.mean(ifs)) / np.std(ifs))
        n_outliers = np.sum(z_scores > 3)
        print(f"{policy}: {n_outliers} influential points")
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