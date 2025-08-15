# Parsimony Implementation Plan

## Analysis Summary

### 1. Compatibility Layer Usage
**Current State:**
- Only imported in ONE place: `base_estimator.py:89`
- Used to convert DiagnosticSuite → legacy formats
- 202 lines of code for backward compatibility

**Usage Pattern:**
```python
# Only path that uses compatibility layer
base_estimator.py → diagnostics_compat.py → IPSDiagnostics/DRDiagnostics
```

**Recommendation:** **DEFER DELETION**
- Keep for 3-6 months to ensure no hidden dependencies
- Add deprecation warning in next release
- Delete after confirming zero usage via telemetry

### 2. DiagnosticSuite Field Usage

**Field Analysis:**
```
Field                 | Used Where           | Keep?
---------------------|---------------------|--------
weight_diagnostics   | Everywhere          | ✅ YES
estimation_summary   | Everywhere          | ✅ YES  
stability           | Runner, display     | ✅ YES
dr_quality          | DR estimators only  | ✅ YES
robust_inference    | Only if requested   | ⚠️ MAYBE
gate_report         | Rarely              | ❌ NO
computation_time    | Never displayed     | ❌ NO
estimator_type      | Compat layer only   | ❌ NO
```

**Actual Usage Count:**
- `weight_diagnostics`: 54 references
- `estimation_summary`: 20 references
- `stability`: 15 references
- `dr_quality`: 14 references
- `robust_inference`: 8 references (mostly in display)
- `gate_report`: 3 references
- `computation_time`: 1 reference (set only)
- `estimator_type`: 3 references (compat only)

**Recommendation:** **SIMPLIFY NOW**
```python
@dataclass
class DiagnosticSuite:
    """Simplified diagnostic suite - only essentials."""
    # Core (always computed)
    weight_diagnostics: Dict[str, WeightMetrics]
    estimation_summary: EstimationSummary
    
    # Optional (computed on demand)
    stability: Optional[StabilityMetrics] = None
    dr_quality: Optional[DRMetrics] = None
    
    # Remove these fields:
    # - robust_inference (too expensive, rarely used)
    # - gate_report (complex, unused)
    # - computation_time (not useful)
    # - estimator_type (only for compat)
```

### 3. DiagnosticRunner Method Analysis

**Method Call Count:**
```
Method                        | Calls | Lines | Inline?
-----------------------------|-------|-------|----------
_compute_weight_diagnostics | 1     | 34    | NO (complex)
_summarize_estimation        | 1     | 39    | NO (complex)
_compute_stability           | 1     | 29    | YES
_compute_dr_quality          | 1     | 28    | YES
_compute_robust_inference    | 1     | 38    | REMOVE
_run_gates                   | 1     | 13    | REMOVE
_suite_to_gate_format        | 1     | 44    | REMOVE
_is_dr_estimator            | 2     | 5     | YES
_has_influence_functions     | 2     | 11    | YES
```

**Recommendation:** **INLINE SELECTIVELY**
- Inline small utility methods (_is_dr_estimator, _has_influence_functions)
- Inline medium methods with single use (_compute_stability, _compute_dr_quality)
- Keep complex core methods (_compute_weight_diagnostics, _summarize_estimation)
- Remove gate-related methods entirely

### 4. Orphaned Module Analysis

**Module Usage:**
```
Module              | Imported By                | Keep?
--------------------|---------------------------|--------
weights.py          | Runner, analyze_dataset   | ✅ YES
stability.py        | Runner                    | ✅ YES
dr.py              | Various DR estimators     | ✅ YES
robust_inference.py | Runner (disabled), tests  | ⚠️ MAYBE
gates.py           | Runner, suite, tests      | ❌ NO
display.py         | analyze_dataset           | ✅ YES
```

**Recommendation:** **REMOVE GATES, QUESTION ROBUST_INFERENCE**
- Delete `gates.py` (785 lines!) - overly complex, unused
- Consider removing `robust_inference.py` (458 lines) - expensive, disabled by default
- Keep others as they provide value

## Implementation Plan

### Phase 1: Quick Wins (1 Day)
**Goal:** Remove obvious dead code

1. **Remove gate-related code:**
   ```python
   # In DiagnosticSuite: remove gate_report field
   # In DiagnosticRunner: remove _run_gates, _suite_to_gate_format
   # In DiagnosticConfig: remove run_gates, gate_config
   # Delete gates.py entirely
   ```

2. **Simplify DiagnosticSuite:**
   ```python
   # Remove: computation_time, estimator_type
   # These add no value
   ```

3. **Inline trivial methods:**
   ```python
   # In DiagnosticRunner.run():
   # Inline _is_dr_estimator (5 lines)
   # Inline _has_influence_functions (11 lines)
   ```

**Impact:** -900 lines, clearer code

### Phase 2: Moderate Simplification (1 Day)
**Goal:** Reduce indirection

1. **Inline single-use methods:**
   ```python
   # Inline _compute_stability into run()
   # Inline _compute_dr_quality into run()
   # Keep as comments: "# Compute stability diagnostics"
   ```

2. **Remove robust_inference (optional):**
   ```python
   # If metrics show <1% usage:
   # - Remove from DiagnosticSuite
   # - Remove _compute_robust_inference
   # - Remove robust_inference.py
   ```

**Impact:** -600 lines, less indirection

### Phase 3: Long-term Consolidation (Future)
**Goal:** Single diagnostic system

**Option A: Enhance IPSDiagnostics**
```python
class IPSDiagnostics:
    # Existing fields...
    stability: Optional[StabilityMetrics] = None
    hill_indices: Dict[str, float] = field(default_factory=dict)
    # Remove tail_ratio_99_5 references
```

**Option B: Minimal DiagnosticSuite**
```python
@dataclass
class Diagnostics:  # Simpler name
    weights: Dict[str, WeightMetrics]
    estimates: EstimationSummary
    stability: Optional[StabilityMetrics] = None
    dr: Optional[DRMetrics] = None
```

**Decision Criteria:**
- If IPSDiagnostics usage > 80%: Enhance it
- If DiagnosticSuite usage > 50%: Simplify it
- Otherwise: Keep both during transition

## Telemetry Plan

Add minimal tracking to understand usage:

```python
# In base_estimator.py
if self._diagnostic_suite:
    logger.debug("TELEMETRY: Using DiagnosticSuite path")
else:
    logger.debug("TELEMETRY: Using legacy diagnostics path")

# In analyze_dataset.py
if hasattr(results, "diagnostic_suite"):
    logger.debug("TELEMETRY: DiagnosticSuite available")
if hasattr(results, "diagnostics"):
    logger.debug("TELEMETRY: Legacy diagnostics available")
```

Run for 1 month, analyze logs.

## Risk Assessment

### Low Risk (Do Now)
- Remove gates.py
- Remove unused fields
- Inline trivial methods

### Medium Risk (Do With Caution)
- Remove robust_inference
- Inline complex methods
- Simplify DiagnosticSuite structure

### High Risk (Defer)
- Delete compatibility layer
- Merge diagnostic systems
- Breaking API changes

## Success Metrics

**Immediate (After Phase 1):**
- Code reduction: -900 lines
- Test coverage: Still 100%
- Performance: No regression

**Short-term (After Phase 2):**
- Code reduction: -1500 lines total
- Clarity: 50% fewer indirection layers
- Maintenance: Easier to understand

**Long-term (After Phase 3):**
- Single diagnostic system
- No compatibility layers
- Clear documentation

## Recommended Immediate Actions

1. **DELETE gates.py** (785 lines)
   - Unused, complex, no value
   - Remove all references

2. **SIMPLIFY DiagnosticSuite**
   - Remove 4 unused fields
   - Keep only essentials

3. **INLINE 2 utility methods**
   - _is_dr_estimator
   - _has_influence_functions

4. **ADD telemetry**
   - Simple DEBUG logs
   - Run for 1 month

5. **DOCUMENT decision**
   - Why we're simplifying
   - What we're keeping

## Do NOT Do (Yet)

1. **Don't delete compatibility layer**
   - Need usage data first
   - No urgency

2. **Don't remove robust_inference yet**
   - Has tests, might be used
   - Need metrics

3. **Don't merge diagnostic systems**
   - Too risky now
   - Wait for usage data

## Final Recommendation

**Start with Phase 1 only.** It's safe, provides immediate value, and removes obvious dead code. The 785-line gates.py module is the clearest win - complex, unused, and easy to remove.

After Phase 1, wait for telemetry data before proceeding. The goal is parsimony, not perfection. Remove what's clearly unused, keep what works, and wait for data on the rest.

**Expected outcome after Phase 1:**
- 900 fewer lines
- Cleaner codebase  
- No functionality loss
- Better maintainability

The key insight: **We can achieve 80% of the simplification with 20% of the risk by focusing on obviously dead code first.**