# CJE Diagnostics Implementation Roadmap

## Overview
This document tracks the implementation of comprehensive diagnostics for the Causal Judge Evaluation (CJE) system, based on Section 9 of the CJE paper. It serves as both a reflection on completed work and a guide for ongoing integration.

## Implementation Philosophy

### Core Principles
1. **Make Assumptions Auditable**: Every statistical assumption should be testable
2. **Fail Transparently**: When assumptions fail, provide clear, actionable diagnostics
3. **Production-First**: Design for real-world use with messy data and dependencies
4. **Modular Components**: Each diagnostic should be independently useful
5. **Paper-Faithful**: Follow the theoretical specifications while adapting to practical constraints

### Key Tensions We've Navigated
- **Theory vs Practice**: Paper assumes perfect conditions; code handles edge cases
- **Memory vs Completeness**: Storing everything for diagnostics vs production constraints
- **Flexibility vs Type Safety**: Research iteration speed vs production reliability
- **Bootstrap Iterations**: Statistical rigor (4000) vs computational cost

## Completed Phases (1-3)

### Phase 1: Core DR Diagnostics ✅
**Goal**: Detect and diagnose fundamental issues with weights and doubly robust estimation

#### What We Built
```python
# cje/utils/diagnostics/weights.py
- hill_tail_index()           # Detect heavy-tailed distributions
- hill_tail_index_stable()     # Multiple k-values for robustness

# cje/utils/diagnostics/dr.py  
- compute_orthogonality_score()    # Validate E[W(R-q)] = 0
- compute_dm_ips_decomposition()   # Separate DM and IPS contributions
```

#### Key Insights
- **Hill Estimator Critical**: α < 2 means variance may be infinite - this is a showstopper
- **Orthogonality as Sanity Check**: Non-zero score indicates misspecification or leakage
- **DM-IPS Balance**: Reveals whether DR is actually helping or if one component dominates

#### Integration Points
- ✅ Automatically computed in `DREstimator.estimate()`
- ✅ Stored in `metadata['orthogonality_scores']` and `metadata['dm_ips_decompositions']`
- ⚠️ Need to surface in visualization dashboards

### Phase 2: Stability & Reliability ✅
**Goal**: Monitor temporal stability and validate calibration assumptions

#### What We Built
```python
# cje/utils/diagnostics/stability.py
- kendall_tau_drift()          # Detect ranking changes over time
- sequential_drift_detection()  # Multi-batch drift monitoring
- reliability_diagram()         # Brier score decomposition
- eif_qq_plot_data()           # Normality testing for inference
- compute_stability_diagnostics() # Integrated analysis
```

#### Key Insights
- **Drift is Gradual**: Kendall τ catches subtle changes before they become critical
- **Calibration != Discrimination**: Brier decomposition separates these concepts
- **Normality Matters**: Heavy-tailed EIFs invalidate standard inference
- **Batch Size Matters**: Too small = noisy; too large = miss local changes

#### Integration Gaps
- ❌ Not yet integrated into main pipeline
- ❌ No automated drift alerts
- ❌ Missing connection to oracle slice refresh triggers

### Phase 3: Robust Inference ✅
**Goal**: Provide valid confidence intervals under real-world conditions

#### What We Built
```python
# cje/utils/diagnostics/robust_inference.py
- stationary_bootstrap_se()     # Time series dependence
- moving_block_bootstrap_se()   # Alternative block method
- cluster_robust_se()           # Session/user clustering
- benjamini_hochberg_correction() # FDR control
- compute_simultaneous_bands()  # Joint confidence regions
- compute_robust_inference()    # Integrated wrapper
```

#### Key Insights
- **4000 Iterations Justified**: Convergence analysis showed 1000 insufficient for tail quantiles
- **Block Length Automation**: First-order ACF works well for moderate dependence
- **Cluster Effects Can Reduce SE**: Counterintuitive but mathematically correct
- **FDR vs FWER**: Benjamini-Hochberg more appropriate than Bonferroni for exploration

#### Implementation Decisions
- No parallelization (simplicity > speed for now)
- Percentile method for CIs (robust to skewness)
- Automatic method selection based on data structure

## Phase 4: Automated Gates ✅

**Goal**: Implement automated quality gates to prevent bad estimates from reaching production

### What We Built
```python
# cje/utils/diagnostics/gates.py
class DiagnosticGate:
    """Base class for diagnostic gates."""
    
    def check(self, diagnostics: Dict) -> GateResult:
        """Returns PASS, WARN, or FAIL with explanation."""
        
class OverlapGate(DiagnosticGate):
    """ESS >= τ and α̂ >= 2"""
    
class JudgeGate(DiagnosticGate):
    """Drift and calibration checks"""
    
class MultiplicityGate(DiagnosticGate):
    """FDR control when |Π| > 5"""

class GateRunner:
    """Orchestrates all gates and produces report."""
    
    def run_all(self, results: EstimationResult) -> GateReport:
        """Run all gates and aggregate results."""
```

### Key Insights
- **Three-Level Status**: PASS/WARN/FAIL provides nuanced feedback
- **Configurable Thresholds**: Users can adjust based on use case
- **Hierarchical Reporting**: Overall status + individual gate details
- **Integration Flexibility**: Opt-in via --gates flag, not forced

### Implementation Decisions
1. **Opt-in by Default**: Gates require explicit --gates flag (safety)
2. **JSON Configuration**: Thresholds via --gate-config JSON
3. **Terminal + Structured**: Both human-readable and machine-parseable output
4. **EstimationResult Integration**: Gates stored in result.gate_report

### Integration Points
- ✅ Added to BaseCJEEstimator with run_gates parameter
- ✅ Integrated into analyze_dataset.py with --gates flag
- ✅ Weight diagnostics properly extracted from IPSDiagnostics
- ✅ Hill tail index computed and stored for heavy-tail detection

## Integration Strategy

### Completed (Phase 4) ✅
1. **EstimationResult Enhanced**:
   - Added `gate_report: Optional[Dict[str, Any]]` field
   - Added `robust_standard_errors` and `robust_confidence_intervals` fields
   - Gates run automatically when `run_gates=True`

2. **analyze_dataset.py Updated**:
   - Added `--gates` flag to enable gate checking
   - Added `--gate-config` for JSON configuration
   - Display function shows color-coded results with recommendations

3. **Diagnostic Dashboard** (Next Priority):
   - Single HTML page with all diagnostics
   - Interactive plots where appropriate
   - Clear pass/warn/fail indicators

### Medium-term (Post-Phase 4)
1. **Streaming Monitoring**:
   - CUSUM/EWMA for continuous drift detection
   - Automatic oracle slice refresh triggers

2. **Negative Controls**:
   - Synthetic null policies for sanity checking
   - Permutation tests for calibration

3. **Adaptive Methods**:
   - Dynamic variance caps based on tail behavior
   - Automatic method switching (IPS → DR) based on diagnostics

## Lessons Learned

### What Worked Well
1. **Modular Design**: Each diagnostic standalone makes testing/debugging easier
2. **Paper Alignment**: Following paper structure provides clear roadmap
3. **Comprehensive Testing**: 42 tests caught many edge cases
4. **Type Annotations**: Mypy caught several subtle bugs

### What Was Challenging
1. **Pydantic Strictness**: Sample/Dataset validation made testing harder
2. **Cross-module Dependencies**: Circular import issues with data models
3. **Bootstrap Performance**: 4000 iterations × multiple policies = slow
4. **Notation Consistency**: Paper uses different notation than codebase

### Unexpected Discoveries
1. **Cluster-robust SE can be smaller**: When clusters reduce variance
2. **Influence functions often missing**: Had to add storage retroactively
3. **Fold assignment complexity**: Multiple sources of fold IDs caused confusion
4. **Type system benefits**: Caught errors that tests missed

## Open Questions

### Technical
1. Should we cache bootstrap distributions for reuse?
2. How to handle missing influence functions gracefully?
3. When should gates be hard stops vs warnings?
4. Optimal default thresholds for each gate?

### Product
1. How much diagnostic detail do users want?
2. Should gates block production deployments?
3. Integration with monitoring/alerting systems?
4. Diagnostic data retention policy?

## Maintenance Notes

### Code Smells to Watch
- Diagnostic computation in hot paths (move to post-processing)
- Duplicated diagnostic logic (consolidate in diagnostic modules)
- Hard-coded thresholds (make configurable)
- Missing error handling in bootstrap loops

### Performance Considerations
- Bootstrap is CPU-bound, not memory-bound
- Consider caching for repeated runs
- Batch diagnostic computations when possible
- Profile before optimizing (likely not bottleneck)

### Testing Strategy
- Unit tests for each diagnostic function
- Integration tests for gate system
- Property-based tests for bootstrap (convergence)
- Regression tests for known edge cases

## Next Steps

### Immediate (Phase 4)
1. Implement gate framework
2. Add CUSUM/EWMA change detection
3. Create unified diagnostic runner
4. Write user-facing documentation

### Future Enhancements
1. Diagnostic visualization dashboard
2. Automated report generation
3. Historical diagnostic tracking
4. Anomaly detection in diagnostic metrics

## References

### Paper Sections
- Section 9.1: Overlap and weight behavior
- Section 9.2: Judge calibration and drift  
- Section 9.3: Doubly robust orthogonality
- Section 9.4: Uncertainty and multiplicity
- Section 9.5: Gates and decision rules

### Key Statistical References
- Hill (1975): Tail index estimation
- Politis & Romano (1994): Stationary bootstrap
- Benjamini & Hochberg (1995): FDR control
- Page (1954): CUSUM charts
- Roberts (1959): EWMA control charts

## Version History
- 2024-01-14: Initial document created after Phase 1-3 completion
- 2025-08-14: Phase 4 completed - Automated gates framework integrated

---

*This is a living document. Update it as the diagnostic system evolves.*