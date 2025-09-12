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
