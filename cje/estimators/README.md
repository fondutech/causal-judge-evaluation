# CJE Estimators

## Overview

The estimators module contains all causal inference estimation methods for unbiased off-policy evaluation of LLMs. These estimators transform logged data and importance weights into reliable policy value estimates with proper uncertainty quantification.

## Estimator Hierarchy

```
BaseCJEEstimator (abstract)
├── RawIPS              # Basic importance sampling
├── CalibratedIPS       # IPS with SIMCal weight calibration
└── DREstimator         # Doubly robust base (abstract)
    ├── DRCPOEstimator  # Basic DR with CPO
    ├── MRDREstimator   # Multiple robust DR
    ├── MRDRTMLEEstimator # MRDR with TMLE
    └── TMLEEstimator   # Targeted maximum likelihood
```

## Core Concepts

### 1. Importance Sampling (IPS)
The foundation of off-policy evaluation. Reweights logged data to estimate performance under new policies using importance weights W = π_target/π_base.

### 2. Weight Calibration (SIMCal)
Stabilizes importance weights through Surrogate-Indexed Monotone Calibration:
- Projects weights to be monotone with judge scores
- Enforces variance constraints to prevent explosion
- Maintains mean-1 property for unbiasedness

### 3. Doubly Robust (DR) Estimation
Combines direct method (outcome model) with IPS correction. Provides two chances to get the estimate right - if either the outcome model OR the weights are correct, DR is consistent.

### 4. Multiple Robustness (MRDR)
Achieves robustness to:
- Outcome model misspecification (via DR)
- Propensity score misspecification (via calibrated weights)
- Both simultaneously (via cross-fitting)

### 5. Targeted Learning (TMLE)
Optimally combines outcome models and importance weights through targeted fluctuation to achieve optimal asymptotic efficiency.

## File Structure

```
estimators/
├── base_estimator.py      # Abstract base with common interface
├── raw_ips.py             # Basic IPS without calibration
├── calibrated_ips.py      # IPS with SIMCal weight calibration
├── dr_base.py             # Doubly robust base class
├── mrdr.py                # Multiple robust DR
├── mrdr_tmle.py           # MRDR with TMLE fluctuation
├── tmle.py                # Standard TMLE
├── outcome_models.py      # Outcome model implementations
└── MRDR_OMEGA_WEIGHTS.md  # Documentation on MRDR weighting schemes
```

## Estimator Selection Guide

### Use **RawIPS** when:
- You have excellent overlap (ESS > 50%)
- You want the simplest baseline
- You don't have judge scores for calibration

### Use **CalibratedIPS** when:
- You have moderate overlap (ESS 20-50%)
- Judge scores are available
- You want variance-stabilized weights
- Fresh draws are not available

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

### 1. Fail-Fast with NaN
When diagnostics indicate catastrophically unreliable estimates (e.g., ESS < 1%), estimators return `NaN` rather than misleading values. This ensures scientific integrity while maintaining pipeline composability.

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

We use percentage-based ESS thresholds (e.g., ESS < 30%) rather than absolute thresholds for several reasons:

1. **Scale Invariance**: Works consistently across datasets of any size without configuration
2. **Measures Overlap Quality**: Low ESS% indicates severe distribution mismatch, not just sample size  
3. **Practical Reliability**: When only 30% of data is effectively used, the estimate is dominated by a small subset, making it practically questionable even if statistically valid
4. **System Simplicity**: One threshold concept that's self-documenting and easy to explain

A 30% ESS threshold means 70% of your data is essentially ignored - this indicates a fundamental problem with policy overlap that more samples won't fix.

### RawIPS Gates
- Refuses if ESS < 1% OR >95% weights near-zero

### CalibratedIPS Gates  
- Refuses if ESS < 30%
- Refuses if >85% weights near-zero
- Refuses if top 5% weight > 30% AND CV > 2.0

### DR Estimator Gates
- Inherits IPS gates
- Warns (but continues) if outcome R² < 0

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
- Data split into k folds
- Each fold gets predictions from model trained on other folds
- Prevents overfitting in outcome models

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
- **SIMCal**: CJE paper (2024)
- **MRDR**: Multiple robustness framework (2024)

## Common Issues

- **Estimates are NaN**: Check ESS in diagnostics. Likely poor overlap - try CalibratedIPS or DR methods.
- **ESS always too low**: Policies may be too different. Consider collecting more diverse base data.
- **DR fails without fresh draws**: All DR methods REQUIRE fresh draws. Generate them first.
- **Different results between runs**: Set random seeds for reproducibility in cross-fitting.

## Summary

The estimators module provides a comprehensive toolkit for causal inference on LLM outputs, from simple importance sampling to sophisticated multiply-robust methods. Each estimator makes different bias-variance tradeoffs, but all follow the same interface and provide transparent diagnostics for reliability assessment.