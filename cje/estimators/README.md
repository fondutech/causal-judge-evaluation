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

# 1. Create sampler with data
sampler = PrecomputedSampler(dataset)

# 2. Initialize estimator
estimator = CalibratedIPS(sampler)

# 3. Fit and estimate
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
- Monotonic regression with judge scores
- No parametric assumptions
- Cross-fitting support

### LinearOutcomeModel  
- Simple linear regression baseline
- Fast and stable
- Good for debugging

### CalibratorBackedOutcomeModel
- Uses the same calibrator as rewards
- Ensures consistency between rewards and predictions
- Default for most DR estimators

### WeightedIsotonicOutcomeModel (MRDR)
- Isotonic regression with importance weighting
- Policy-specific models
- Configurable omega weights (see MRDR_OMEGA_WEIGHTS.md)

## Fresh Draws Integration

DR estimators can incorporate fresh draws (new responses from target policies):

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

### Variance Computation
Standard errors computed from influence functions:
```python
se = np.std(influence_functions, ddof=1) / np.sqrt(n)
```

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