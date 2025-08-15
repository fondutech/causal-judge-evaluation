# Compatibility Layer Removal Plan

## Executive Summary

The compatibility layer (`diagnostics_compat.py`) is a 202-line bridge between the new DiagnosticSuite and legacy IPSDiagnostics/DRDiagnostics. It's only imported in one place but has downstream dependencies. This plan outlines a safe, phased removal strategy.

## Current State Analysis

### Dependency Graph
```
DiagnosticSuite (new)
    ↓
base_estimator.py (line 89-110)
    ↓
diagnostics_compat.py (202 lines)
    ↓
IPSDiagnostics/DRDiagnostics (legacy)
    ↓
Used by:
- analyze_dataset.py (6 references)
- visualization/dr_dashboards.py (1 reference)  
- tests/ (25+ references)
- base_estimator.py gates (5 references)
```

### Usage Locations

1. **Primary Users of result.diagnostics:**
   - `analyze_dataset.py`: Display and visualization
   - `dr_dashboards.py`: DR-specific visualizations
   - Tests: Validation of diagnostic output

2. **What They Access:**
   ```python
   # Common patterns:
   results.diagnostics.worst_if_tail_ratio
   results.diagnostics.ess_per_policy
   results.diagnostics.n_samples_valid
   results.diagnostics.dr_diagnostics_per_policy
   results.diagnostics.method
   ```

3. **Dual Support Already Exists:**
   - analyze_dataset.py ALREADY checks for diagnostic_suite first!
   - Falls back to legacy diagnostics if not found

## Removal Strategy

### Phase 0: Add Telemetry (Week 1)
**Goal:** Understand actual usage patterns

```python
# In base_estimator.py:87
if not hasattr(result, "diagnostics") or result.diagnostics is None:
    # Log telemetry
    import logging
    logger = logging.getLogger(__name__)
    logger.info("TELEMETRY: Creating legacy diagnostics from DiagnosticSuite")
    
    # Add deprecation warning
    import warnings
    warnings.warn(
        "Legacy diagnostics (result.diagnostics) are deprecated. "
        "Use result.diagnostic_suite instead. "
        "Legacy support will be removed in v2.0.0",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Continue with compatibility layer...
```

### Phase 1: Parallel Support (Weeks 2-4)
**Goal:** Ensure both paths work correctly

1. **Always populate both fields:**
   ```python
   # In base_estimator.py
   result.diagnostic_suite = self._diagnostic_suite  # New
   result.diagnostics = create_legacy_diagnostics()  # Legacy (with warning)
   ```

2. **Update documentation:**
   ```python
   """
   EstimationResult.diagnostics - DEPRECATED, use diagnostic_suite
   EstimationResult.diagnostic_suite - New unified diagnostics
   """
   ```

3. **Add migration guide:**
   ```markdown
   ## Migrating from result.diagnostics to result.diagnostic_suite
   
   Old: result.diagnostics.worst_if_tail_ratio
   New: result.diagnostic_suite.dr_quality.worst_if_tail_ratio
   
   Old: result.diagnostics.ess_per_policy
   New: result.diagnostic_suite.weight_diagnostics[policy].ess
   ```

### Phase 2: Update Core Users (Weeks 5-8)
**Goal:** Migrate primary consumers

1. **Update analyze_dataset.py:**
   ```python
   # Already partially done! Just remove fallback:
   if hasattr(results, "diagnostic_suite") and results.diagnostic_suite:
       # Use diagnostic_suite (no fallback needed)
   ```

2. **Update visualization/dr_dashboards.py:**
   ```python
   # Old:
   if isinstance(estimation_result.diagnostics, DRDiagnostics):
       dr_diags = estimation_result.diagnostics.dr_diagnostics_per_policy
   
   # New:
   if estimation_result.diagnostic_suite and estimation_result.diagnostic_suite.dr_quality:
       dr_diags = estimation_result.diagnostic_suite.dr_quality
   ```

3. **Update tests gradually:**
   - Keep old tests working during transition
   - Add new tests for diagnostic_suite
   - Mark old tests as deprecated

### Phase 3: Deprecation Period (Weeks 9-16)
**Goal:** Give users time to migrate

1. **Make compatibility opt-in:**
   ```python
   # In base_estimator.__init__
   def __init__(self, ..., legacy_diagnostics=False):
       self.legacy_diagnostics = legacy_diagnostics
   
   # In fit_and_estimate
   if self.legacy_diagnostics:
       # Create legacy format with louder warning
   ```

2. **Track usage:**
   - Monitor telemetry for legacy access
   - Reach out to users still using old format

3. **Update all examples and docs**

### Phase 4: Removal (Week 17+)
**Goal:** Complete removal

1. **Delete compatibility layer:**
   ```bash
   rm cje/data/diagnostics_compat.py
   ```

2. **Remove legacy code from base_estimator.py:**
   ```python
   # Delete lines 86-110 (compatibility code)
   ```

3. **Clean up imports:**
   ```python
   # Remove from base_estimator.py
   # from ..data.diagnostics_compat import ...
   ```

4. **Update any remaining references**

## Migration Helpers

### For Users: Accessor Functions
Create helper functions for common patterns:

```python
# In cje/diagnostics/helpers.py
def get_ess_per_policy(result):
    """Get ESS from either diagnostic format."""
    if hasattr(result, 'diagnostic_suite') and result.diagnostic_suite:
        return {
            policy: metrics.ess 
            for policy, metrics in result.diagnostic_suite.weight_diagnostics.items()
        }
    elif hasattr(result, 'diagnostics'):
        return result.diagnostics.ess_per_policy
    return {}

def get_worst_tail_ratio(result):
    """Get worst tail ratio from either format."""
    if hasattr(result, 'diagnostic_suite') and result.diagnostic_suite:
        if result.diagnostic_suite.dr_quality:
            return result.diagnostic_suite.dr_quality.worst_if_tail_ratio
    elif hasattr(result, 'diagnostics'):
        return getattr(result.diagnostics, 'worst_if_tail_ratio', 0.0)
    return 0.0
```

### For Developers: Automated Migration Script
```python
#!/usr/bin/env python
"""Migrate code from result.diagnostics to result.diagnostic_suite."""

import re
import sys
from pathlib import Path

MIGRATIONS = [
    (r'result\.diagnostics\.ess_per_policy', 
     'get_ess_per_policy(result)'),
    (r'result\.diagnostics\.worst_if_tail_ratio',
     'get_worst_tail_ratio(result)'),
    # Add more patterns...
]

def migrate_file(filepath):
    """Apply migrations to a single file."""
    content = filepath.read_text()
    original = content
    
    for pattern, replacement in MIGRATIONS:
        content = re.sub(pattern, replacement, content)
    
    if content != original:
        filepath.write_text(content)
        print(f"✓ Migrated {filepath}")
        return True
    return False

if __name__ == "__main__":
    path = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    migrated = 0
    for filepath in path.rglob("*.py"):
        if migrate_file(filepath):
            migrated += 1
    print(f"Migrated {migrated} files")
```

## Risk Mitigation

### Backward Compatibility Guarantee
```python
# In base_estimator.py during transition
try:
    # Always try to populate both
    result.diagnostic_suite = suite
    result.diagnostics = create_legacy(suite)
except Exception as e:
    logger.error(f"Failed to create legacy diagnostics: {e}")
    # But don't fail the whole computation
```

### Feature Flags
```python
# In settings or environment
CJE_LEGACY_DIAGNOSTICS = os.getenv("CJE_LEGACY_DIAGNOSTICS", "true")

if CJE_LEGACY_DIAGNOSTICS == "true":
    # Keep compatibility layer
```

### Rollback Plan
1. Git tag before removal: `git tag pre-compat-removal`
2. Keep compatibility code in a branch: `compat-layer-archive`
3. Can restore with: `git cherry-pick <commit>`

## Success Metrics

### Phase 0-1 (Telemetry)
- ✅ Deprecation warnings visible to users
- ✅ Telemetry shows usage patterns
- ✅ No functionality breaks

### Phase 2 (Migration)
- ✅ Core users migrated (analyze_dataset.py, visualization)
- ✅ 50% of tests updated
- ✅ Documentation updated

### Phase 3 (Deprecation)
- ✅ <10% of runs use legacy format
- ✅ All examples use new format
- ✅ No critical user complaints

### Phase 4 (Removal)
- ✅ Compatibility layer deleted
- ✅ 202 lines removed
- ✅ All tests pass
- ✅ No user impact

## Timeline

| Week | Phase | Action | Risk |
|------|-------|--------|------|
| 1 | 0 | Add telemetry & warnings | None |
| 2-4 | 1 | Parallel support | Low |
| 5-8 | 2 | Migrate core users | Medium |
| 9-16 | 3 | Deprecation period | Low |
| 17+ | 4 | Complete removal | Medium |

## Decision Points

### Week 4: Continue or Abort?
- If telemetry shows >80% legacy usage → extend timeline
- If telemetry shows <20% legacy usage → accelerate

### Week 8: Ready for Deprecation?
- If core users migrated → proceed
- If issues found → extend Phase 2

### Week 16: Ready for Removal?
- If legacy usage <5% → proceed with removal
- If legacy usage >5% → extend deprecation

## Alternative: Permanent Compatibility

If removal proves too risky, consider:

1. **Make it a permanent feature:**
   ```python
   class EstimationResult:
       @property
       def diagnostics(self):
           """Legacy accessor for backward compatibility."""
           if not hasattr(self, '_legacy_cache'):
               self._legacy_cache = create_legacy(self.diagnostic_suite)
           return self._legacy_cache
   ```

2. **Minimize maintenance:**
   - Keep compatibility layer but don't enhance
   - Document as "legacy but supported"
   - 202 lines is not terrible if it prevents breakage

## Recommendation

**Start with Phase 0 immediately:**
1. Add telemetry today (5 minutes)
2. Add deprecation warning (5 minutes)
3. Run for 2-4 weeks
4. Analyze usage data
5. Decide on timeline based on actual usage

**Key insight:** The compatibility layer is well-isolated (one import site) and serves a clear purpose. Removal is feasible but not urgent. The 202 lines are a small price for backward compatibility during transition.

**Success looks like:**
- Smooth transition with no user disruption
- Clear communication via deprecation warnings
- Data-driven timeline based on actual usage
- Option to keep if removal proves too disruptive

The compatibility layer is technical debt, but it's *managed* technical debt that enables a safe transition.