# Diagnostics Integration Technical Specification

## Current State Analysis

### What's Built
We have three layers of diagnostic capabilities implemented:
1. **Weight diagnostics** (`weights.py`): ESS, tail indices, concentration
2. **DR diagnostics** (`dr.py`): Orthogonality, decomposition, policy-specific
3. **Stability diagnostics** (`stability.py`): Drift, calibration, normality
4. **Robust inference** (`robust_inference.py`): Bootstrap, FDR, clusters

### What's Connected
- ✅ Weight diagnostics → `CalibratedIPS`
- ✅ DR diagnostics → `DREstimator` 
- ✅ Hill tail index → Weight diagnostics
- ⚠️ Stability diagnostics → Standalone only
- ⚠️ Robust inference → Standalone only

### What's Missing
- ❌ Automated gates that block/warn
- ❌ Stability monitoring in main pipeline
- ❌ Robust SEs as default option
- ❌ Diagnostic visualization
- ❌ Historical tracking

## Integration Architecture

### Data Flow
```
Dataset → Calibration → Estimation → Diagnostics → Gates → Results
                ↓                           ↓                    ↓
            Oracle Slice              Stability Check      Visualization
```

### Key Integration Points

#### 1. EstimationResult Enhancement
```python
@dataclass
class EstimationResult:
    # Existing fields
    estimates: np.ndarray
    standard_errors: np.ndarray
    
    # Add robust inference
    robust_standard_errors: Optional[np.ndarray] = None
    robust_confidence_intervals: Optional[List[Tuple[float, float]]] = None
    
    # Add stability metrics
    stability_diagnostics: Optional[Dict[str, Any]] = None
    
    # Add gate results
    gate_report: Optional[GateReport] = None
```

#### 2. Estimator Base Class Updates
```python
class BaseCJEEstimator:
    def __init__(self, sampler, run_diagnostics=True, run_gates=False):
        self.run_diagnostics = run_diagnostics
        self.run_gates = run_gates
        
    def estimate(self) -> EstimationResult:
        # Core estimation
        result = self._estimate_core()
        
        # Add diagnostics
        if self.run_diagnostics:
            result = self._add_diagnostics(result)
            
        # Run gates
        if self.run_gates:
            result.gate_report = self._run_gates(result)
            
        return result
```

#### 3. Configuration Schema
```yaml
# cje_config.yaml
diagnostics:
  enabled: true
  
  stability:
    check_drift: true
    drift_window: 1000
    kendall_tau_threshold: 0.7
    
  robustness:
    use_robust_se: true
    bootstrap_iterations: 4000
    cluster_field: "session_id"
    
  gates:
    enabled: true
    overlap:
      ess_min: 1000
      tail_index_min: 2.0
    judge:
      drift_threshold: 0.05
      calibration_ece_max: 0.1
    multiplicity:
      fdr_level: 0.05
      min_policies: 5
```

## Module-Specific Integration

### cje/experiments/arena_10k_simplified/analyze_dataset.py

#### Current Code Pattern
```python
# Load and calibrate
dataset = load_dataset_from_jsonl(args.data)
calibrated_dataset, cal_result = calibrate_dataset(...)

# Run estimation
sampler = PrecomputedSampler(calibrated_dataset)
estimator = create_estimator(args.estimator, sampler, ...)
results = estimator.fit_and_estimate()
```

#### Enhanced Pattern
```python
# Load and calibrate
dataset = load_dataset_from_jsonl(args.data)

# Add stability check
if args.check_stability:
    stability = compute_stability_diagnostics(dataset)
    if stability['drift_detection']['has_drift']:
        logger.warning(f"Drift detected: {stability['drift_detection']}")

calibrated_dataset, cal_result = calibrate_dataset(...)

# Run estimation with diagnostics
sampler = PrecomputedSampler(calibrated_dataset)
estimator = create_estimator(
    args.estimator, 
    sampler,
    run_diagnostics=True,
    run_gates=args.gates
)
results = estimator.fit_and_estimate()

# Add robust inference if requested
if args.robust_se:
    results = add_robust_inference(results, method=args.robust_method)

# Display gate results
if results.gate_report:
    print_gate_report(results.gate_report)
```

### cje/visualization/

#### New Diagnostic Dashboard
```python
# cje/visualization/diagnostic_dashboard.py

def create_diagnostic_dashboard(
    results: EstimationResult,
    output_path: str = "diagnostics.html"
) -> None:
    """Create comprehensive diagnostic dashboard."""
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[
            "Weight Distribution",
            "Tail Behavior", 
            "Orthogonality Check",
            "Drift Detection",
            "Calibration",
            "EIF Q-Q Plot",
            "DM vs IPS",
            "Gate Status",
            "Summary Stats"
        ]
    )
    
    # Add each diagnostic plot
    add_weight_plot(fig, results, row=1, col=1)
    add_tail_plot(fig, results, row=1, col=2)
    # ... etc
    
    # Save as interactive HTML
    fig.write_html(output_path)
```

## Testing Strategy

### Unit Test Updates
```python
# cje/tests/test_integration.py

def test_diagnostics_in_estimation():
    """Test that diagnostics are computed during estimation."""
    dataset = create_test_dataset()
    sampler = PrecomputedSampler(dataset)
    
    estimator = CalibratedIPS(sampler, run_diagnostics=True)
    results = estimator.fit_and_estimate()
    
    assert results.diagnostics is not None
    assert 'tail_index' in results.metadata
    
def test_gates_block_on_failure():
    """Test that gates prevent bad estimates."""
    dataset = create_pathological_dataset()  # Very heavy tails
    sampler = PrecomputedSampler(dataset)
    
    estimator = CalibratedIPS(sampler, run_gates=True)
    results = estimator.fit_and_estimate()
    
    assert results.gate_report.overall_status == "FAIL"
    assert "tail_index" in results.gate_report.failures
```

### Integration Test Scenarios
1. **Drift During Estimation**: Simulate temporal data with drift
2. **Heavy Tails + Gates**: Ensure gates catch pathological weights
3. **Multiple Policies + FDR**: Verify FDR control works end-to-end
4. **Robust SE Comparison**: Check robust vs classical SEs

## Performance Considerations

### Diagnostic Overhead
- Weight diagnostics: ~1ms per policy
- DR diagnostics: ~5ms per policy  
- Stability check: ~100ms for 10K samples
- Bootstrap (4000): ~2-5s per policy
- Gates: ~10ms total

### Optimization Opportunities
1. **Lazy Computation**: Only compute expensive diagnostics if gates require
2. **Caching**: Store bootstrap distributions for reuse
3. **Batching**: Compute all policy diagnostics together
4. **Parallelization**: Bootstrap iterations are embarrassingly parallel

## Migration Plan

### Phase 4a: Gate Framework (Week 1)
1. Implement base gate classes
2. Create standard gates (overlap, judge, multiplicity)
3. Add gate runner with configurable thresholds
4. Unit tests for each gate

### Phase 4b: Integration (Week 2)
1. Update EstimationResult dataclass
2. Modify base estimator for diagnostics/gates
3. Update analyze_dataset.py script
4. Integration tests

### Phase 4c: Visualization (Week 3)
1. Create diagnostic dashboard
2. Add to existing visualization module
3. Export functions for reports
4. Documentation

### Phase 4d: Deployment (Week 4)
1. Update experiment scripts
2. Add configuration files
3. Performance profiling
4. User documentation

## Backward Compatibility

### Breaking Changes
- None planned - all additions are optional

### Deprecation Path
```python
# Old way (still works)
results = estimator.fit_and_estimate()

# New way (with diagnostics)
results = estimator.fit_and_estimate(
    run_diagnostics=True,
    run_gates=True
)
```

### Configuration Migration
- Default: diagnostics ON, gates OFF
- Explicit opt-in for gates (safety)
- Environment variable override: `CJE_RUN_GATES=1`

## Risk Assessment

### Technical Risks
1. **Performance Impact**: Bootstrap could slow pipeline
   - Mitigation: Make optional, add caching
   
2. **Memory Usage**: Storing all diagnostics
   - Mitigation: Lazy evaluation, cleanup old data
   
3. **Numerical Stability**: Bootstrap edge cases
   - Mitigation: Extensive testing, fallbacks

### Product Risks
1. **Over-conservative Gates**: Too many false warnings
   - Mitigation: Tunable thresholds, override mechanism
   
2. **User Confusion**: Too many diagnostics
   - Mitigation: Clear documentation, sensible defaults

## Success Metrics

### Quantitative
- Gate catch rate for known bad scenarios: >95%
- False positive rate for gates: <5%
- Performance overhead: <10% for typical workload
- Test coverage: >90% for diagnostic code

### Qualitative
- Users understand gate failures
- Diagnostics help debug issues
- System catches problems before production
- Clear upgrade path from current system

## Open API Questions

1. Should `run_diagnostics` be a constructor arg or method arg?
2. Should gates be hierarchical (fail fast) or comprehensive?
3. How to handle partial gate failures?
4. Should we version gate thresholds?

## Next Steps

1. Review this spec with team
2. Finalize API decisions
3. Begin Phase 4a implementation
4. Create user-facing documentation

---

*Last Updated: 2024-01-14*