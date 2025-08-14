# CJE Diagnostic Integration Plan

## Executive Summary

After completing Phase 5 Week 1, we have a powerful unified diagnostic system (DiagnosticSuite) that isn't actually being used. The old diagnostic system still runs everywhere, causing double computation and confusion. This plan addresses the integration debt before adding new features.

## Current State Assessment

### 🔴 Critical Issues
1. **Double Computation**: Diagnostics computed in both old IPSDiagnostics and new DiagnosticSuite
2. **Disabled by Default**: Stability diagnostics (drift detection) never run unless manually enabled
3. **Data Loss**: Influence functions stored in wrong location, causing robust inference to fail
4. **Orphaned System**: DiagnosticSuite exists but isn't used in the main pipeline

### 📊 By the Numbers
- **58** remaining references to deprecated `tail_ratio`
- **18** total references to new `DiagnosticSuite` (all in diagnostic module itself)
- **2x** performance overhead from double computation
- **0%** of users see stability diagnostics (disabled by default)

## Integration Plan

### Phase 1: Fix Critical Issues (Day 1-2)

#### 1.1 Remove Double Computation in CalibratedIPS
**Problem**: `_build_diagnostics()` computes old IPSDiagnostics, then `fit_and_estimate()` computes DiagnosticSuite.

**Solution**:
```python
# In CalibratedIPS.estimate():
if not self.run_diagnostics:
    # Skip old diagnostic computation
    result.diagnostics = None
else:
    # Let BaseCJEEstimator.fit_and_estimate() handle it
    pass
```

**Files to modify**:
- `cje/estimators/calibrated_ips.py` - Remove `_build_diagnostics()` call
- `cje/estimators/dr_base.py` - Same pattern
- `cje/estimators/raw_ips.py` - Update to match

#### 1.2 Enable Stability Diagnostics by Default
**Problem**: `check_stability=False` means drift detection never runs.

**Solution**:
```python
@dataclass
class DiagnosticConfig:
    check_stability: bool = True  # Changed from False
    check_dr_quality: bool = True
    compute_robust_se: bool = False  # Keep expensive option off
```

**Files to modify**:
- `cje/diagnostics/runner.py` - Change default

#### 1.3 Fix Influence Function Storage
**Problem**: DR stores in `metadata["dr_influence"]`, but runner looks in `result.influence_functions`.

**Solution**:
```python
# In DR estimators, store in BOTH locations:
result.influence_functions = influence_functions
result.metadata["dr_influence"] = influence_functions  # For backward compat
```

**Files to modify**:
- `cje/estimators/dr_base.py` - Store in result.influence_functions
- `cje/estimators/mrdr.py` - Same pattern
- `cje/estimators/tmle.py` - Same pattern

### Phase 2: Bridge Old and New (Day 3-4)

#### 2.1 Create Compatibility Layer
**Goal**: Make IPSDiagnostics a view into DiagnosticSuite.

```python
class IPSDiagnostics:
    """Legacy diagnostic format - now backed by DiagnosticSuite."""
    
    def __init__(self, suite: DiagnosticSuite):
        self._suite = suite
        # Populate legacy fields from suite
        self._populate_from_suite()
    
    @classmethod
    def from_suite(cls, suite: DiagnosticSuite) -> "IPSDiagnostics":
        """Create legacy diagnostics from new suite."""
        ...
```

**Files to create**:
- `cje/data/diagnostics_compat.py` - Compatibility layer

#### 2.2 Update EstimationResult
**Goal**: Populate both old and new diagnostic formats.

```python
# In BaseCJEEstimator.fit_and_estimate():
if self.run_diagnostics:
    suite = runner.run(self, result)
    result.diagnostic_suite = suite
    
    # Populate legacy format for backward compatibility
    if self._is_ips_estimator():
        result.diagnostics = IPSDiagnostics.from_suite(suite)
    elif self._is_dr_estimator():
        result.diagnostics = DRDiagnostics.from_suite(suite)
```

### Phase 3: Update Main Pipeline (Day 5-6)

#### 3.1 Modernize analyze_dataset.py
**Goal**: Make it the reference implementation using DiagnosticSuite.

```python
def analyze_with_estimator(estimator_name, sampler, args):
    # Create diagnostic config from args
    diag_config = DiagnosticConfig(
        check_stability=not args.no_stability,
        compute_robust_se=args.robust_se,
        run_gates=args.gates,
    )
    
    # Create estimator with config
    estimator = create_estimator(
        estimator_name, 
        sampler,
        diagnostic_config=diag_config
    )
    
    # Run estimation
    results = estimator.fit_and_estimate()
    
    # Display using new system
    if results.diagnostic_suite:
        display_diagnostic_suite(
            results.diagnostic_suite,
            verbosity=args.verbosity
        )
```

**Files to modify**:
- `cje/experiments/arena_10k_simplified/analyze_dataset.py`

#### 3.2 Add CLI Arguments
```python
parser.add_argument(
    "--verbosity",
    choices=["quiet", "normal", "detailed"],
    default="normal",
    help="Diagnostic output verbosity"
)
parser.add_argument(
    "--no-stability",
    action="store_true",
    help="Skip stability diagnostics (drift detection)"
)
parser.add_argument(
    "--robust-se",
    action="store_true", 
    help="Compute robust standard errors (slow)"
)
```

### Phase 4: Update Visualization (Day 7-8)

#### 4.1 Create Adapter for Existing Visualizations
**Goal**: Make existing plots work with DiagnosticSuite without rewriting everything.

```python
def plot_weight_diagnostics_from_suite(suite: DiagnosticSuite):
    """Adapter to use existing plot with new suite."""
    # Convert suite to format expected by old plot
    old_format = {
        "ess_per_policy": {
            p: m.ess for p, m in suite.weight_diagnostics.items()
        },
        "tail_ratio_per_policy": {  # Compute for compatibility
            p: _compute_tail_ratio_from_hill(m.hill_index) 
            for p, m in suite.weight_diagnostics.items()
        }
    }
    return plot_weight_diagnostics(old_format)
```

**Files to create**:
- `cje/visualization/diagnostic_adapters.py`

#### 4.2 Create New Unified Dashboard
```python
def create_diagnostic_dashboard(suite: DiagnosticSuite, output_path: str):
    """Create comprehensive diagnostic dashboard from suite."""
    fig = make_subplots(rows=3, cols=3, ...)
    
    # Row 1: Weights
    add_weight_distribution(fig, suite, row=1, col=1)
    add_ess_comparison(fig, suite, row=1, col=2) 
    add_tail_analysis(fig, suite, row=1, col=3)
    
    # Row 2: Quality
    add_stability_plot(fig, suite, row=2, col=1)
    add_orthogonality_plot(fig, suite, row=2, col=2)
    add_gate_summary(fig, suite, row=2, col=3)
    
    # Row 3: Inference
    add_bootstrap_comparison(fig, suite, row=3, col=1)
    add_recommendations(fig, suite, row=3, col=2)
    add_quality_score(fig, suite, row=3, col=3)
    
    fig.write_html(output_path)
```

**Files to create**:
- `cje/visualization/diagnostic_dashboard.py`

### Phase 5: Testing and Validation (Day 9-10)

#### 5.1 Integration Tests
```python
def test_no_double_computation():
    """Ensure diagnostics computed only once."""
    with patch('cje.diagnostics.runner.DiagnosticRunner.run') as mock_run:
        estimator = CalibratedIPS(sampler)
        result = estimator.fit_and_estimate()
        assert mock_run.call_count == 1  # Not 2!

def test_influence_functions_available():
    """Ensure influence functions accessible for robust inference."""
    estimator = DRCPOEstimator(sampler)
    result = estimator.fit_and_estimate()
    assert result.diagnostic_suite.robust_inference is not None
```

**Files to create**:
- `cje/tests/test_diagnostic_integration.py`

#### 5.2 Performance Benchmarks
```python
def benchmark_diagnostic_performance():
    """Compare old vs new diagnostic computation time."""
    # Time old system
    # Time new system
    # Assert new is not slower
```

### Phase 6: Documentation and Migration (Day 11-12)

#### 6.1 Update Documentation
- Update README with new diagnostic system
- Create migration guide for existing code
- Document new CLI arguments

#### 6.2 Deprecation Warnings
```python
def tail_weight_ratio(...):
    warnings.warn(
        "tail_weight_ratio is deprecated. Use hill_tail_index instead.",
        DeprecationWarning,
        stacklevel=2
    )
```

## Success Criteria

### Functional
- [ ] No double computation of diagnostics
- [ ] Stability diagnostics run by default
- [ ] Influence functions available for robust inference
- [ ] analyze_dataset.py uses DiagnosticSuite
- [ ] Visualizations work with new format

### Performance
- [ ] Diagnostic computation ≤ 5s for 10K samples
- [ ] No performance regression vs old system
- [ ] Memory usage stable (no leaks)

### Quality
- [ ] All tests pass
- [ ] Mypy passes
- [ ] Black formatting passes
- [ ] No deprecation warnings in main flow

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking existing code | Compatibility layer maintains old interface |
| Performance regression | Benchmark before/after each change |
| Missing edge cases | Comprehensive test suite |
| User confusion | Clear migration guide with examples |

## Timeline

### Week 1 (Current)
- Day 1-2: Fix critical issues
- Day 3-4: Bridge old and new
- Day 5-6: Update main pipeline

### Week 2
- Day 7-8: Update visualization
- Day 9-10: Testing and validation
- Day 11-12: Documentation

### Week 3
- Remove deprecated code
- Final optimization
- Release preparation

## Decision Points

1. **Should we maintain backward compatibility indefinitely?**
   - Recommendation: Maintain for 2 releases, then remove
   
2. **Should expensive diagnostics (bootstrap) be opt-in or opt-out?**
   - Recommendation: Opt-in (current approach is correct)
   
3. **Should we show all diagnostics or progressive disclosure?**
   - Recommendation: Progressive based on verbosity level

## Next Steps

1. **Immediate**: Fix the three critical issues
2. **This week**: Complete Phase 1-3 of integration plan
3. **Next week**: Complete Phase 4-6
4. **Future**: Consider diagnostic persistence and trending

## Appendix: Code Locations

### Key Files to Modify
```
cje/estimators/calibrated_ips.py    # Remove double computation
cje/estimators/dr_base.py           # Fix influence functions
cje/diagnostics/runner.py           # Enable stability by default
cje/experiments/arena_10k_simplified/analyze_dataset.py  # Reference implementation
```

### New Files to Create
```
cje/data/diagnostics_compat.py      # Compatibility layer
cje/visualization/diagnostic_adapters.py  # Visualization adapters  
cje/visualization/diagnostic_dashboard.py # New dashboard
cje/tests/test_diagnostic_integration.py  # Integration tests
```

### Files to Eventually Deprecate
```
cje/utils/diagnostics/display.py    # Replaced by cje/diagnostics/display.py
Multiple references to tail_ratio_*  # Replaced by hill_index
```

---

*This plan prioritizes integration over new features. A fully integrated simple system beats a partially integrated complex system.*