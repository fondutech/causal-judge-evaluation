# Stacked SIMCal Implementation Plan

## Executive Summary

Replace the current single-direction SIMCal with a **stacked approach** that combines {baseline, increasing, decreasing} candidates via convex optimization to minimize out-of-fold (OOF) influence function variance. This achieves oracle-style efficiency while maintaining all guarantees (mean-one, non-negativity, ESS/variance caps).

## Core Design Principles

1. **Parsimony**: Remove ALL legacy selection heuristics (L2, single-IF). Go straight to stacking.
2. **Unbiased**: Use OOF influence functions with fold-wise centering for selection.
3. **Unified constraints**: Apply ESS/variance caps via a single post-stack γ-blend.
4. **Default to best**: Stack by default in CalibratedIPS, no fallback paths.
5. **Clean API**: Minimal changes to external interface.

## What Gets Removed

### 1. From `cje/calibration/simcal.py`:
- `direction` parameter (no more "auto", "increasing", "decreasing")
- `tie_break` parameter (no more "ess" vs "var")
- `select_direction_by` parameter (no more "l2", "ips_if", "dr_if")
- All the conditional logic for direction selection
- The complex nested if-else for different selection methods

### 2. From `cje/estimators/calibrated_ips.py`:
- `select_direction_by` parameter
- All the fallback logic (DR -> IPS -> L2)
- The complex residual computation logic for direction selection
- The `_select_best_direction_legacy` method (already removed)

## What Gets Added

### 1. New Stacked SIMCal (`cje/calibration/simcal.py` - complete replacement)

```python
@dataclass
class SimcalConfig:
    """Configuration for stacked SIMCal calibration.
    
    Args:
        ess_floor: Minimum ESS as fraction of n (e.g., 0.2 => ESS >= 0.2 * n)
        var_cap: Maximum allowed variance of calibrated weights
        epsilon: Small constant for numerical stability
        include_baseline: Whether to include raw weights in the stack (default True)
        ridge_lambda: Ridge regularization for covariance matrix (default 1e-8)
        n_folds: Number of folds for OOF if fold_ids not provided (default 5)
    """
    ess_floor: Optional[float] = 0.2
    var_cap: Optional[float] = None
    epsilon: float = 1e-9
    include_baseline: bool = True
    ridge_lambda: float = 1e-8
    n_folds: int = 5

class SIMCalibrator:
    """Stacked Score-Indexed Monotone Calibrator.
    
    Combines {baseline, increasing, decreasing} candidates to minimize
    OOF influence function variance, then applies uniform blending to
    meet ESS/variance constraints.
    """
```

### 2. Simplified CalibratedIPS

```python
class CalibratedIPS(BaseCJEEstimator):
    def __init__(
        self,
        sampler: PrecomputedSampler,
        clip_weight: Optional[float] = None,
        ess_floor: Optional[float] = 0.2,
        var_cap: Optional[float] = None,
        calibrator: Optional[Any] = None,  # For DR residuals if available
    ):
        # No more select_direction_by parameter
```

## Implementation Steps

### Phase 1: Core Stacked Implementation
1. **Backup current simcal.py** for reference
2. **Replace SimcalConfig** with simplified version (no direction params)
3. **Rewrite SIMCalibrator.transform()** to:
   - Build candidates: {baseline?, increasing, decreasing}
   - Compute OOF influence matrix
   - Solve quadratic program on simplex
   - Apply single γ-blend for constraints
4. **Remove all legacy selection code**

### Phase 2: Integrate with CalibratedIPS
1. **Simplify constructor** - remove select_direction_by
2. **Simplify fit()** - always use stacked approach
3. **Update diagnostics** to include mixture weights

### Phase 3: Testing & Validation
1. **Update existing tests** to remove direction-based assertions
2. **Add stacking-specific tests**:
   - Mixture weights sum to 1
   - OOF variance reduction
   - Constraint satisfaction
3. **Benchmark against legacy** on test data

### Phase 4: Documentation
1. **Update docstrings** to reflect stacking
2. **Update CLAUDE.md** architectural notes
3. **Update README.md** examples

## Key Implementation Details

### OOF Influence Functions

For each candidate weight vector w and fold f:
```python
# IPS influence (if no residuals)
IF = w * rewards - theta_f  # theta_f = mean on training folds

# DR influence (if residuals available)
IF = w * residuals - theta_f  # residuals = rewards - g_oof(scores)
```

### Quadratic Program

Minimize variance of stacked influence function:
```
min_π  π^T Σ π
s.t.   π ≥ 0, 1^T π = 1

where Σ_ij = Cov(IF_i, IF_j) across OOF samples
```

### Active Set Solution (for K=3 candidates)
```python
def solve_simplex_qp(Sigma):
    # Start with all candidates
    # If any π_i < 0, drop most negative
    # Re-solve on reduced set
    # Repeat until feasible
```

### Post-Stack Blending
```python
# Single γ for all constraints
γ = max(γ_ess, γ_var)  # where each γ satisfies respective constraint
w_final = 1 + (1-γ)(w_stacked - 1)
```

## Migration Path

### For Users
- **No API changes** for basic usage
- `CalibratedIPS(sampler)` works as before but uses stacking internally
- Results should be similar or better (lower variance)

### For Advanced Users
- Can control stacking via `SimcalConfig`:
  - `include_baseline=False` to exclude raw weights
  - `ridge_lambda` to adjust regularization
  - `n_folds` for OOF splitting

### Backward Compatibility
- Keep old SimcalConfig fields as deprecated (ignored)
- Log warning if old parameters used
- Remove in next major version

## Success Metrics

1. **Variance Reduction**: Stacked weights should have ≤ variance of best single candidate
2. **Constraint Satisfaction**: ESS/variance caps always met
3. **Computation Time**: < 2x current implementation
4. **Test Coverage**: 100% of new code paths
5. **Documentation**: Clear explanation of stacking benefit

## Risk Mitigation

### Numerical Stability
- Ridge regularization on covariance matrix
- Fallback to baseline if QP degenerate
- Careful handling of near-zero weights

### Performance
- Cache candidate computations
- Vectorize OOF centering
- Consider numba for QP solver if needed

### Validation
- Compare with legacy on existing test suite
- Synthetic tests with known optimal solutions
- Real data benchmarks from experiments/

## Timeline

- **Day 1**: Core implementation (Phase 1)
- **Day 2**: Integration & testing (Phase 2-3)
- **Day 3**: Documentation & benchmarking (Phase 4)

## Open Questions

1. **Shrinkage to baseline**: Add small (e.g., 5%) shrink for stability?
   - **Recommendation**: Yes, but make configurable

2. **Enforce monotonicity after stacking**: Optional projection?
   - **Recommendation**: No, keep it simple. Stack can be non-monotonic.

3. **Default ridge_lambda**: How much regularization?
   - **Recommendation**: 1e-8 * trace(Σ) / K

4. **Fold assignment**: Require explicit fold_ids or auto-generate?
   - **Recommendation**: Auto-generate with stable seed if not provided

## Decision Log

- **Include baseline by default**: YES - provides safety, QP can zero it out
- **Single γ-blend**: YES - simpler than per-candidate blending
- **OOF for selection**: YES - unbiased, matches inference
- **Remove all legacy**: YES - parsimony over backward compatibility
- **Make default immediately**: YES - no feature flags, just switch

## Next Steps

1. Review and approve this plan
2. Create backup branch for legacy code
3. Begin Phase 1 implementation
4. Set up benchmark comparisons

---

**Approval**: [ ] Ready to proceed with implementation