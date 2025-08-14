# Diagnostic System Integration Analysis

## Current State Assessment

### What We've Built (Phases 1-4)

```
cje/utils/diagnostics/
├── weights.py         # ESS, tail indices, concentration
├── dr.py             # Orthogonality, DM-IPS decomposition  
├── stability.py      # Drift, calibration, normality
├── robust_inference.py # Bootstrap, FDR, cluster SEs
├── gates.py          # Automated quality gates
└── display.py        # Formatting utilities
```

### Architectural Review

#### ✅ What's Working Well
1. **Modular Components**: Each diagnostic is independently usable
2. **Paper Fidelity**: Close alignment with Section 9 specifications
3. **Type Safety**: Proper typing with Pydantic models
4. **Test Coverage**: 40+ tests across diagnostic modules

#### ⚠️ Parsimony Issues

**1. Redundant Computations**
- `tail_ratio_99_5` (percentile-based) vs `hill_tail_index` (Hill estimator)
  - Both measure heavy tails - pick ONE
- Weight diagnostics computed in 3 places:
  - `compute_weight_diagnostics()` 
  - `CalibratedIPS._build_diagnostics()`
  - Gate framework re-extracts from IPSDiagnostics

**2. Over-Engineering**
- Two bootstrap methods (stationary + moving block) when one suffices
- Five gates when 2-3 core gates would cover 90% of cases
- Normality gate is off by default (not essential)

**3. Scattered State**
- Diagnostics stored in:
  - `result.diagnostics` (IPSDiagnostics/DRDiagnostics objects)
  - `result.metadata` (tail indices, orthogonality scores)
  - `estimator._diagnostics` (internal caches)
  - Separate gate reports

#### ❌ Integration Gaps

**1. Orphaned Modules**
- `stability.py`: Not called anywhere in main pipeline
- `robust_inference.py`: No automatic usage, requires manual invocation
- Phase 2 & 3 diagnostics compute but don't surface to users

**2. Missing Connections**
- Visualization module unaware of new diagnostics
- No unified diagnostic dashboard
- Gates can't access stability diagnostics (they're never computed!)
- No diagnostic history/persistence

**3. User Experience Gaps**
- Diagnostics computed but buried in metadata
- Gates warn but don't guide fixes
- No progressive disclosure (beginner vs expert views)

## Refactoring for Parsimony

### Principle: "Make it simple, then make it powerful"

### 1. Consolidate Redundant Metrics
```python
# BEFORE: Two tail measures
tail_ratio = np.percentile(weights, 99.5) / np.percentile(weights, 50)
hill_index = hill_tail_index(weights)

# AFTER: One canonical measure
tail_index = hill_tail_index(weights)  # More theoretically grounded
# Interpretation: α < 2 means infinite variance (critical)
```

### 2. Unify Diagnostic Storage
```python
@dataclass
class DiagnosticSuite:
    """Single source of truth for all diagnostics."""
    
    # Core metrics (always computed)
    weights: WeightDiagnostics
    
    # Optional advanced diagnostics
    stability: Optional[StabilityDiagnostics] = None
    dr_quality: Optional[DRQualityDiagnostics] = None
    inference: Optional[InferenceDiagnostics] = None
    
    # Gate results
    gates: Optional[GateReport] = None
    
    def to_summary(self) -> str:
        """Human-readable summary."""
    
    def to_dashboard(self) -> Dict:
        """Data for visualization."""
```

### 3. Simplify Gate System
```python
# Core gates only (80/20 rule)
ESSENTIAL_GATES = [
    OverlapGate,      # ESS + tail behavior
    StabilityGate,    # Drift + calibration (merge Judge + Normality)
    MultiplicityGate, # FDR when |Π| > 5
]
```

## Integration Plan

### Phase 5: Unification & Simplification

#### Step 1: Create Unified Diagnostic API (Week 1)
```python
class DiagnosticRunner:
    """Single entry point for all diagnostics."""
    
    def __init__(self, config: DiagnosticConfig):
        self.config = config
        
    def run_all(self, result: EstimationResult) -> DiagnosticSuite:
        """Compute all enabled diagnostics."""
        suite = DiagnosticSuite()
        
        # Always compute core diagnostics
        suite.weights = self._compute_weight_diagnostics(result)
        
        # Conditionally compute advanced diagnostics
        if self.config.check_stability:
            suite.stability = self._compute_stability(result)
            
        if self.config.check_dr and isinstance(result.diagnostics, DRDiagnostics):
            suite.dr_quality = self._compute_dr_quality(result)
            
        if self.config.compute_robust_se:
            suite.inference = self._compute_robust_inference(result)
            
        # Run gates if requested
        if self.config.run_gates:
            suite.gates = self._run_gates(suite)
            
        return suite
```

#### Step 2: Integrate into Estimators (Week 1)
```python
class BaseCJEEstimator:
    def fit_and_estimate(self) -> EstimationResult:
        # Core estimation
        result = self._estimate_core()
        
        # Single diagnostic call
        if self.run_diagnostics:
            runner = DiagnosticRunner(self.diagnostic_config)
            result.diagnostic_suite = runner.run_all(result)
        
        return result
```

#### Step 3: Connect Visualization (Week 2)
```python
def create_diagnostic_dashboard(result: EstimationResult) -> None:
    """Generate comprehensive diagnostic visualization."""
    
    if not result.diagnostic_suite:
        raise ValueError("No diagnostics available")
    
    suite = result.diagnostic_suite
    
    # Create multi-panel dashboard
    fig = make_subplots(rows=2, cols=3, ...)
    
    # Panel 1: Weight distribution & tail behavior
    plot_weight_diagnostics(fig, suite.weights, row=1, col=1)
    
    # Panel 2: Stability over time (if available)
    if suite.stability:
        plot_stability_metrics(fig, suite.stability, row=1, col=2)
    
    # Panel 3: Gate status summary
    if suite.gates:
        plot_gate_summary(fig, suite.gates, row=1, col=3)
    
    # Panel 4: Policy comparison with robust CIs
    if suite.inference:
        plot_robust_comparison(fig, suite.inference, row=2, col=1)
    
    # Export
    fig.write_html("diagnostics.html")
```

#### Step 4: Update analyze_dataset.py (Week 2)
```python
# Simplified interface
parser.add_argument(
    "--diagnostics",
    choices=["basic", "advanced", "full"],
    default="basic",
    help="Diagnostic level"
)

# Map to configuration
DIAGNOSTIC_CONFIGS = {
    "basic": DiagnosticConfig(
        check_stability=False,
        compute_robust_se=False,
        run_gates=True,
        gate_thresholds="conservative"
    ),
    "advanced": DiagnosticConfig(
        check_stability=True,
        compute_robust_se=True,
        run_gates=True,
        gate_thresholds="moderate"
    ),
    "full": DiagnosticConfig(
        check_stability=True,
        compute_robust_se=True,
        run_gates=True,
        gate_thresholds="strict",
        save_history=True
    )
}
```

### Phase 6: User Experience (Week 3)

#### Progressive Disclosure
```python
def display_diagnostics(suite: DiagnosticSuite, verbosity: str = "normal"):
    if verbosity == "minimal":
        # Just show pass/fail
        print(f"✅ Diagnostics: {suite.gates.overall_status}")
        
    elif verbosity == "normal":
        # Show key metrics
        print(suite.to_summary())
        
    elif verbosity == "detailed":
        # Full diagnostic report
        print(suite.to_detailed_report())
        
        # Actionable recommendations
        if suite.gates.has_failures():
            print("\n🔧 Recommended Actions:")
            for action in suite.get_recommended_actions():
                print(f"  • {action}")
```

#### Diagnostic History
```python
class DiagnosticHistory:
    """Track diagnostics over time."""
    
    def append(self, suite: DiagnosticSuite, timestamp: datetime):
        """Add diagnostic snapshot."""
        
    def detect_trends(self) -> List[Trend]:
        """Identify concerning patterns."""
        
    def export_report(self) -> str:
        """Generate time-series diagnostic report."""
```

### Phase 7: Documentation & Examples (Week 4)

#### User Guide Structure
1. **Quick Start**: Basic diagnostics in 3 commands
2. **Interpretation Guide**: What each metric means
3. **Troubleshooting**: Common issues and fixes
4. **Advanced Usage**: Custom gates and thresholds
5. **API Reference**: Complete diagnostic API

## Migration Strategy

### Deprecation Path
```python
# Phase 1: Add warnings (v2.1)
if args.use_old_diagnostics:
    warnings.warn(
        "Old diagnostic API deprecated. Use --diagnostics flag instead.",
        DeprecationWarning
    )

# Phase 2: Remove old code (v3.0)
# Delete redundant diagnostic computations
# Remove scattered metadata storage
```

### Backward Compatibility
- Keep existing diagnostic fields in EstimationResult
- Populate them from DiagnosticSuite for compatibility
- Gradually migrate visualization to use DiagnosticSuite

## Success Metrics

### Quantitative
- **Code Reduction**: Remove 30% of diagnostic code through consolidation
- **Performance**: All diagnostics complete in < 5s for 10K samples
- **Coverage**: 95% of estimation runs include basic diagnostics
- **Actionability**: 80% of gate failures have automated fix suggestions

### Qualitative  
- **Discoverability**: Users find and use diagnostics without documentation
- **Interpretability**: Non-experts understand diagnostic outputs
- **Trust**: Teams rely on gates for production decisions

## Open Design Questions

1. **Should diagnostics be computed lazily or eagerly?**
   - Lazy: Better performance, but may miss issues
   - Eager: Comprehensive, but slower
   - Hybrid: Core eager, advanced lazy?

2. **How to handle diagnostic failures in production?**
   - Hard fail: Block deployment
   - Soft fail: Log and alert
   - Graduated: Severity-based response

3. **Diagnostic configuration management?**
   - YAML files for reproducibility
   - Environment variables for CI/CD
   - Code-based for version control

## Next Actions

### Immediate (This Week)
1. [ ] Create DiagnosticSuite dataclass
2. [ ] Implement DiagnosticRunner 
3. [ ] Remove redundant tail_ratio_99_5
4. [ ] Consolidate weight diagnostic computations

### Short-term (Next 2 Weeks)
1. [ ] Connect stability diagnostics to main pipeline
2. [ ] Create unified dashboard
3. [ ] Implement diagnostic history
4. [ ] Write user guide

### Long-term (Next Month)
1. [ ] Add diagnostic caching
2. [ ] Implement trend detection
3. [ ] Create diagnostic CI/CD integration
4. [ ] Build monitoring dashboard

---

*The goal: Make diagnostics so simple and useful that they become invisible infrastructure, not visible complexity.*