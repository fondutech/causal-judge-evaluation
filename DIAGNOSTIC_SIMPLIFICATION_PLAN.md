# Diagnostic System Simplification Plan

## Current State Assessment

### The Problem
The diagnostic system has grown organically with multiple overlapping layers:
1. **Raw dictionaries** (original implementation)
2. **IPSDiagnostics/DRDiagnostics objects** (structured diagnostics)
3. **Metadata fields** (for DR decompositions)
4. **DiagnosticSuite** (latest abstraction attempt)

This violates CLAUDE.md principles:
- "Do One Thing Well" - diagnostics are doing too many things
- "YAGNI" - DiagnosticSuite adds abstraction we don't need
- "Simple, correct, and maintainable - not clever"

### What's Actually Working
- IPSDiagnostics works well for IPS-based estimators
- DRDiagnostics structure is sound (but not properly populated)
- The display functions can handle both diagnostic types

### What's Not Working
- DR estimators return metadata dicts instead of DRDiagnostics
- DiagnosticSuite adds complexity without solving real problems
- Display code has 3+ code paths for essentially the same data
- DR decomposition values show as zeros (not populated)

## Recommended Approach: Simplify Back to Basics

### Phase 1: Fix What We Have (Priority)

#### 1.1 Make DR Diagnostics Actually Work
**Problem**: DR estimators show zeros for decomposition values
**Solution**: Properly populate DRDiagnostics in dr_base.py

```python
# In DREstimator.estimate()
diagnostics = DRDiagnostics(
    estimator_type="DR",
    method=self.method,
    n_samples_total=len(self.sampler.dataset.samples),
    n_samples_valid=self.sampler.n_valid_samples,
    # Actually populate these:
    dm_ips_decompositions={
        policy: {
            "dm_contribution": dm_mean,
            "ips_augmentation": ips_corr,
            "total": estimate
        }
    },
    orthogonality_scores={policy: ortho_score},
    outcome_model_r2={policy: r2_value},
    # Include weight diagnostics from IPS
    weight_ess=self.ips_estimator.get_diagnostics().weight_ess,
    ess_per_policy=self.ips_estimator.get_diagnostics().ess_per_policy,
)
result.diagnostics = diagnostics
```

#### 1.2 Remove DiagnosticSuite
**Rationale**: It's an unnecessary abstraction that adds complexity
**Action**: 
- Remove cje/diagnostics/suite.py
- Remove cje/diagnostics/runner.py  
- Remove references from base_estimator.py
- Clean up imports

#### 1.3 Consolidate Display Logic
**Current**: 3 code paths in display_weight_diagnostics
**Target**: 1 simple path

```python
def display_weight_diagnostics(estimator, sampler, dataset, args):
    """Display weight diagnostics."""
    print("\n5. Weight diagnostics:")
    
    # Get diagnostics from the appropriate source
    if hasattr(estimator, 'ips_estimator'):
        # DR estimator - get from IPS
        diagnostics = estimator.ips_estimator.get_diagnostics()
    else:
        # Direct IPS estimator
        diagnostics = estimator.get_diagnostics()
    
    if diagnostics and hasattr(diagnostics, 'ess_per_policy'):
        # Display using the diagnostics object
        print(create_weight_summary_table(diagnostics))
        # Check for issues...
    else:
        # Fallback: compute manually (shouldn't happen)
        print("Weight diagnostics not available")
```

### Phase 2: Standardize Interfaces

#### 2.1 Consistent get_diagnostics()
Every estimator returns its appropriate diagnostic type:
- IPS-based → IPSDiagnostics
- DR-based → DRDiagnostics
- Never plain dicts

#### 2.2 Clear Ownership
- Each estimator owns its diagnostic creation
- No external runners or suites
- Diagnostics computed during estimate()

### Phase 3: Clean Up Technical Debt

#### 3.1 Remove Legacy Paths
- Remove dictionary-based diagnostic passing
- Remove metadata diagnostic fields
- Keep only IPSDiagnostics and DRDiagnostics

#### 3.2 Simplify analyze_dataset.py
- Extract estimator creation to factory
- Reduce nested conditionals
- Single path for each diagnostic type

## Implementation Order

### Week 1: Fix Critical Issues
1. **Fix DR diagnostics population** (2 hours)
   - Modify dr_base.py to properly fill DRDiagnostics
   - Test with dr-cpo estimator
   
2. **Remove DiagnosticSuite** (1 hour)
   - Delete files
   - Clean up imports
   - Update base_estimator.py

### Week 2: Consolidate Display
3. **Simplify display_weight_diagnostics** (2 hours)
   - Reduce to single code path
   - Remove duplication
   
4. **Clean up analyze_dataset.py** (3 hours)
   - Extract constants
   - Simplify estimator setup
   - Reduce nesting

### Week 3: Remove Legacy Code
5. **Remove dictionary diagnostics** (2 hours)
   - Update all estimators
   - Remove legacy display paths
   
6. **Documentation and testing** (2 hours)
   - Update docstrings
   - Add tests for diagnostic objects

## Success Metrics

1. **Correctness**: DR diagnostics show actual values (not zeros)
2. **Simplicity**: Display functions < 50 lines each
3. **Clarity**: Each estimator has ONE diagnostic type
4. **Maintainability**: No duplicate code paths

## Risks and Mitigations

**Risk**: Breaking existing functionality
**Mitigation**: Make changes incrementally with tests

**Risk**: Missing edge cases
**Mitigation**: Keep fallback paths initially, remove after validation

**Risk**: User confusion during transition
**Mitigation**: Clear error messages, no silent failures

## Non-Goals

- Creating new abstractions
- Unifying all diagnostics into one type
- Adding new features
- Performance optimization

## Principles to Follow

From CLAUDE.md:
- "Do One Thing Well" - Each diagnostic type serves one estimator family
- "Explicit is Better than Implicit" - Clear diagnostic ownership
- "Fail Fast and Clearly" - No magic fallbacks
- "YAGNI" - Don't build what isn't needed

## Alternative Considered: Full DiagnosticSuite Commitment

**Why Not**: 
- Requires rewriting all estimators
- Adds abstraction without clear benefit
- Violates YAGNI principle
- The problem isn't lack of structure, it's lack of data population

The existing IPSDiagnostics/DRDiagnostics separation is actually good design - it acknowledges that IPS and DR estimators have fundamentally different diagnostic needs. The issue is implementation, not architecture.

## Next Immediate Step

1. Fix DR diagnostics population in dr_base.py
2. Test with `poetry run python analyze_dataset.py --data "data copy"/cje_dataset.jsonl --estimator dr-cpo --oracle-coverage .5`
3. Verify decomposition values show correctly
4. Then proceed with DiagnosticSuite removal

This plan follows the principle of "make it work, make it right, make it fast" - we're still at "make it work" for DR diagnostics.