# Phase 5: Diagnostic System Integration & Simplification

## Executive Summary

After implementing Phases 1-4, we have powerful diagnostic capabilities but poor integration. Phase 5 focuses on **simplification, unification, and user experience**.

## Key Problems to Solve

1. **Redundancy**: Tail behavior measured 2+ ways, weight diagnostics computed 3x
2. **Orphaned Code**: Stability (Phase 2) and robust inference (Phase 3) unused
3. **Scattered State**: Diagnostics in metadata, objects, and internal caches  
4. **Poor UX**: Diagnostics computed but not surfaced effectively

## Simplification Decisions

### What to Keep (Core)
- ✅ Hill tail index (remove tail_ratio_99_5)
- ✅ ESS as primary overlap metric
- ✅ 3 essential gates (Overlap, Stability, Multiplicity)
- ✅ Stationary bootstrap only (remove moving block)
- ✅ Single source of truth (DiagnosticSuite)

### What to Remove/Deprecate
- ❌ tail_ratio_99_5 (redundant with Hill)
- ❌ Moving block bootstrap (stationary is sufficient)
- ❌ NormalityGate (rarely useful in practice)
- ❌ Duplicate weight diagnostic computations
- ❌ Scattered metadata storage

## Implementation Roadmap

### Week 1: Core Refactoring

#### Day 1-2: Create Unified Diagnostic API
```python
# cje/diagnostics/suite.py
@dataclass
class DiagnosticSuite:
    """Single source of truth for all diagnostics."""
    
    # Always computed
    weight_diagnostics: Dict[str, WeightMetrics]
    estimation_summary: EstimationSummary
    
    # Conditionally computed
    stability: Optional[StabilityMetrics] = None
    dr_quality: Optional[DRMetrics] = None
    robust_inference: Optional[RobustInference] = None
    gate_report: Optional[GateReport] = None
    
    @property
    def has_issues(self) -> bool:
        """Quick check if any issues detected."""
        if self.gate_report:
            return self.gate_report.overall_status != GateStatus.PASS
        # Fallback to heuristics
        min_ess = min(w.ess for w in self.weight_diagnostics.values())
        return min_ess < 100
    
    def get_recommendations(self) -> List[str]:
        """Actionable recommendations based on diagnostics."""
        recs = []
        
        # Check ESS
        for policy, metrics in self.weight_diagnostics.items():
            if metrics.ess < 100:
                recs.append(f"⚠️ {policy}: Increase sample size (ESS={metrics.ess:.0f})")
        
        # Check tail behavior
        for policy, metrics in self.weight_diagnostics.items():
            if metrics.hill_index and metrics.hill_index < 2:
                recs.append(f"⚠️ {policy}: Heavy tails detected (α={metrics.hill_index:.2f})")
                recs.append(f"   → Consider variance regularization or DR estimation")
        
        # Check stability
        if self.stability and self.stability.has_drift:
            recs.append("⚠️ Judge drift detected")
            recs.append("   → Refresh oracle labels or retrain judge")
            
        return recs
```

#### Day 3-4: Consolidate Diagnostic Computation
```python
# cje/diagnostics/runner.py
class DiagnosticRunner:
    """Centralized diagnostic computation."""
    
    def __init__(self, config: Optional[DiagnosticConfig] = None):
        self.config = config or DiagnosticConfig()
    
    def run(self, 
            estimator: BaseCJEEstimator,
            result: EstimationResult) -> DiagnosticSuite:
        """Run all configured diagnostics."""
        
        suite = DiagnosticSuite(
            weight_diagnostics=self._compute_weights(estimator, result),
            estimation_summary=self._summarize_estimation(result)
        )
        
        # Conditional diagnostics
        if self.config.check_stability:
            suite.stability = compute_stability_diagnostics(
                estimator.sampler.dataset
            )
        
        if self.config.compute_robust_se and result.influence_functions:
            suite.robust_inference = compute_robust_inference(
                result.estimates,
                result.influence_functions,
                method="stationary_bootstrap",
                n_bootstrap=self.config.n_bootstrap
            )
        
        if self.config.run_gates:
            suite.gate_report = self._run_gates(suite)
            
        return suite
    
    def _compute_weights(self, estimator, result):
        """Single computation of weight diagnostics."""
        diagnostics = {}
        
        for policy in estimator.sampler.target_policies:
            weights = estimator.get_weights(policy)
            if weights is not None:
                diagnostics[policy] = WeightMetrics(
                    ess=effective_sample_size(weights),
                    max_weight=float(np.max(weights)),
                    hill_index=hill_tail_index(weights),
                    cv=float(np.std(weights) / np.mean(weights))
                )
        
        return diagnostics
```

#### Day 5: Update Estimators
```python
# cje/estimators/base_estimator.py
class BaseCJEEstimator:
    def __init__(self, sampler, diagnostic_config=None):
        self.sampler = sampler
        self.diagnostic_config = diagnostic_config or DiagnosticConfig()
        self._diagnostic_suite: Optional[DiagnosticSuite] = None
    
    def fit_and_estimate(self) -> EstimationResult:
        self.fit()
        result = self.estimate()
        
        # Single diagnostic call
        if self.diagnostic_config.enabled:
            runner = DiagnosticRunner(self.diagnostic_config)
            self._diagnostic_suite = runner.run(self, result)
            
            # Store in result for backward compatibility
            result.diagnostic_suite = self._diagnostic_suite
            
            # Populate legacy fields
            self._populate_legacy_diagnostics(result)
        
        return result
```

### Week 2: Visualization & UX

#### Day 1-2: Create Unified Dashboard
```python
# cje/visualization/diagnostic_dashboard.py
def create_diagnostic_dashboard(
    suite: DiagnosticSuite,
    output_path: str = "diagnostics.html"
) -> None:
    """Create interactive diagnostic dashboard."""
    
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[
            "Weight Distribution",
            "Tail Behavior (Hill Index)",
            "ESS by Policy",
            "Stability Check",
            "Gate Summary", 
            "Policy Estimates",
            "Robust vs Classical SEs",
            "Recommendations",
            "Quality Score"
        ],
        specs=[[...]]  # Proper spec configuration
    )
    
    # Row 1: Weight diagnostics
    _add_weight_distribution(fig, suite, row=1, col=1)
    _add_tail_analysis(fig, suite, row=1, col=2)
    _add_ess_comparison(fig, suite, row=1, col=3)
    
    # Row 2: Quality metrics
    if suite.stability:
        _add_stability_plot(fig, suite.stability, row=2, col=1)
    
    if suite.gate_report:
        _add_gate_summary(fig, suite.gate_report, row=2, col=2)
    
    _add_policy_comparison(fig, suite, row=2, col=3)
    
    # Row 3: Inference and recommendations
    if suite.robust_inference:
        _add_se_comparison(fig, suite, row=3, col=1)
    
    _add_recommendations(fig, suite.get_recommendations(), row=3, col=2)
    _add_quality_score(fig, suite, row=3, col=3)
    
    # Style and export
    fig.update_layout(
        title="CJE Diagnostic Dashboard",
        height=1200,
        showlegend=True
    )
    
    fig.write_html(output_path)
    print(f"📊 Dashboard saved to {output_path}")
```

#### Day 3-4: Improve analyze_dataset.py UX
```python
# Simplified interface with progressive disclosure
def display_diagnostics(suite: DiagnosticSuite, args):
    """Display diagnostics based on verbosity level."""
    
    if args.verbosity == "quiet":
        # Just show overall status
        status = "✅" if not suite.has_issues else "⚠️"
        print(f"\nDiagnostics: {status}")
        
    elif args.verbosity == "normal":
        # Show summary
        print("\n" + "="*60)
        print("DIAGNOSTIC SUMMARY")
        print("="*60)
        
        # Weight diagnostics
        print("\n📊 Weight Diagnostics:")
        for policy, metrics in suite.weight_diagnostics.items():
            status = "✅" if metrics.ess > 100 else "⚠️"
            print(f"  {status} {policy}: ESS={metrics.ess:.0f}, α={metrics.hill_index:.2f}")
        
        # Gate summary if available
        if suite.gate_report:
            print(f"\n🚦 Gates: {suite.gate_report.summary}")
        
        # Recommendations
        recs = suite.get_recommendations()
        if recs:
            print("\n💡 Recommendations:")
            for rec in recs[:3]:  # Top 3
                print(f"  {rec}")
                
    elif args.verbosity == "detailed":
        # Full report
        print(suite.to_detailed_report())
        
        # All recommendations
        print("\n" + "="*60)
        print("ACTIONABLE RECOMMENDATIONS")
        print("="*60)
        for rec in suite.get_recommendations():
            print(rec)
```

#### Day 5: Add Diagnostic Export
```python
def export_diagnostics(suite: DiagnosticSuite, args):
    """Export diagnostics in multiple formats."""
    
    if args.export_format == "json":
        # Machine-readable
        with open(args.export_path, "w") as f:
            json.dump(suite.to_dict(), f, indent=2)
            
    elif args.export_format == "html":
        # Interactive dashboard
        create_diagnostic_dashboard(suite, args.export_path)
        
    elif args.export_format == "pdf":
        # Static report
        create_pdf_report(suite, args.export_path)
```

### Week 3: Testing & Documentation

#### Day 1-2: Comprehensive Testing
```python
# cje/tests/test_diagnostic_integration.py
def test_diagnostic_suite_completeness():
    """Ensure all diagnostics are captured."""
    dataset = create_test_dataset()
    sampler = PrecomputedSampler(dataset)
    estimator = CalibratedIPS(sampler)
    result = estimator.fit_and_estimate()
    
    assert result.diagnostic_suite is not None
    suite = result.diagnostic_suite
    
    # Check all core diagnostics present
    assert len(suite.weight_diagnostics) == len(sampler.target_policies)
    assert all(m.ess > 0 for m in suite.weight_diagnostics.values())
    assert all(m.hill_index is not None for m in suite.weight_diagnostics.values())

def test_backward_compatibility():
    """Ensure legacy code still works."""
    # Old way should still work
    assert result.diagnostics is not None  # IPSDiagnostics
    assert "tail_indices" in result.metadata  # Legacy metadata
```

#### Day 3-5: User Documentation
```markdown
# CJE Diagnostics User Guide

## Quick Start

```bash
# Basic diagnostics (default)
python analyze_dataset.py --data data.jsonl

# Advanced diagnostics with gates
python analyze_dataset.py --data data.jsonl --diagnostics advanced

# Export dashboard
python analyze_dataset.py --data data.jsonl --export-diagnostics diagnostics.html
```

## Understanding Your Diagnostics

### 🟢 Green Flags (Good)
- ESS > 500 per policy
- Hill index α > 2.5
- No drift detected
- All gates PASS

### 🟡 Yellow Flags (Monitor)
- ESS 100-500
- Hill index 2.0-2.5
- Minor drift (Δτ < 0.05)
- Some gates WARN

### 🔴 Red Flags (Action Required)
- ESS < 100
- Hill index α < 2.0 (infinite variance!)
- Major drift (Δτ > 0.1)
- Any gate FAIL

## Common Issues & Solutions

### Issue: "ESS too low"
**Cause**: Insufficient overlap between policies
**Solutions**:
1. Increase sample size
2. Use less extreme target policies
3. Switch to DR estimation (more robust)

### Issue: "Heavy tails detected (α < 2)"
**Cause**: Extreme importance weights
**Solutions**:
1. Enable variance capping (`--var-cap 10`)
2. Use calibrated weights (default)
3. Consider MRDR for multiple policies
```

### Week 4: Deprecation & Migration

#### Deprecation Plan
```python
# Add deprecation warnings
def compute_weight_diagnostics_old(weights):
    warnings.warn(
        "compute_weight_diagnostics is deprecated. "
        "Use DiagnosticRunner instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Still works but delegates to new code
    return DiagnosticRunner()._compute_single_weight_metric(weights)
```

#### Migration Guide
```markdown
# Migrating to DiagnosticSuite

## Old Way
```python
# Scattered diagnostics
weight_diag = compute_weight_diagnostics(weights)
tail_idx = hill_tail_index(weights)
results.metadata["tail_indices"] = {...}
```

## New Way
```python
# Unified suite
suite = result.diagnostic_suite
metrics = suite.weight_diagnostics[policy]
# Everything in one place!
```
```

## Success Criteria

### Week 1
- [ ] DiagnosticSuite implemented and tested
- [ ] DiagnosticRunner consolidates all computations
- [ ] Estimators use new diagnostic system

### Week 2  
- [ ] Dashboard visualization working
- [ ] Progressive disclosure in CLI
- [ ] Export functionality complete

### Week 3
- [ ] 100% backward compatibility
- [ ] User guide published
- [ ] Migration guide available

### Week 4
- [ ] Deprecation warnings in place
- [ ] Performance benchmarked (< 5s for 10K samples)
- [ ] Team trained on new system

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking existing code | Maintain backward compatibility layer |
| Performance regression | Lazy computation for expensive diagnostics |
| User confusion | Progressive disclosure + documentation |
| Missing edge cases | Comprehensive test suite |

## Long-term Vision

**6 Months**: Diagnostics become invisible infrastructure
- Automatic issue detection and remediation
- Historical tracking with trend analysis  
- Integration with monitoring/alerting
- Self-tuning thresholds

**1 Year**: Diagnostics drive method selection
- Automatically choose IPS vs DR based on diagnostics
- Dynamic variance capping based on tail behavior
- Adaptive oracle refresh based on drift detection

---

*Make diagnostics so good they become boring - reliable infrastructure that just works.*