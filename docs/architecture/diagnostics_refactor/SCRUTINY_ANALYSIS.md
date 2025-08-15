# Scrutiny Analysis: What to Remove vs Keep

## Executive Summary

After careful analysis, here are the clear recommendations:

| Component | Lines | Verdict | Action |
|-----------|-------|---------|--------|
| gates.py | 785 | **DELETE** | Remove entirely |
| gate_report field | ~20 | **DELETE** | Remove from DiagnosticSuite |
| computation_time | ~5 | **DELETE** | Remove field |
| estimator_type | ~5 | **DELETE** | Remove field |
| robust_inference | ~100 | **KEEP** | Has value, just disabled |
| Single-use methods | ~150 | **KEEP** | Improve readability |

**Total Safe Removal: ~815 lines**

## Detailed Analysis

### 1. gates.py (785 lines) - DELETE ✂️

**What it does:**
- Implements automated stop/ship decisions based on thresholds
- Complex hierarchy of gate classes (BaseGate, WeightGate, OrthogonalityGate, etc.)
- Supposed to help with "production readiness"

**Usage analysis:**
```python
# Actual usage:
run_gates = False  # DEFAULT everywhere
--gates  # CLI flag exists but never documented/used
```

**Problems:**
1. **Never enabled by default** - `run_gates=False` everywhere
2. **No documentation** - Users don't know it exists
3. **Over-engineered** - 785 lines for simple threshold checks
4. **No clear value** - What problem does it solve?
5. **Complex configuration** - Requires detailed gate_config dict

**Recommendation: DELETE ENTIRELY**
- Zero impact on users (they don't know it exists)
- Removes 785 lines of untested complexity
- Threshold checks can be done simpler if needed

### 2. gate_report Field - DELETE ✂️

**Current usage:**
```python
# In DiagnosticSuite
gate_report: Optional[GateReport] = None  # Never set unless gates enabled

# In display
if suite.gate_report:  # Never true in practice
    lines.append(suite.gate_report.format_terminal())
```

**Recommendation: DELETE**
- Field is None 99.9% of the time
- Depends on gates.py which we're deleting
- Saves field + display logic

### 3. computation_time Field - DELETE ✂️

**Current usage:**
```python
# Set once
suite.computation_time = time.time() - start_time

# Displayed if present
if suite.computation_time:
    lines.append(f"Computation Time: {suite.computation_time:.2f}s")
```

**Analysis:**
- Diagnostics are fast (<1s typically)
- Not actionable information
- Users never asked for timing
- Clutters output

**Recommendation: DELETE**
- No value to users
- Can add back if someone asks

### 4. estimator_type Field - DELETE ✂️

**Current usage:**
```python
# Set from class name
estimator_type=estimator.__class__.__name__

# Used in:
- Display (redundant, users know what they ran)
- Compatibility layer (which we're keeping anyway)
```

**Analysis:**
- Users know which estimator they used
- Redundant with other context
- Only used for compatibility conversion

**Recommendation: DELETE**
- Pass directly to compatibility layer if needed
- Don't store in suite

### 5. robust_inference Field - KEEP BUT IMPROVE ✅

**Current usage:**
```python
# Disabled by default but has real value
compute_robust_se: bool = False  # Keep expensive bootstrap off by default

# When enabled, provides:
- Bootstrap standard errors
- Bootstrap confidence intervals  
- FDR-corrected p-values
- Multiple testing correction
```

**Analysis:**
- **Has real statistical value** (unlike gates)
- **Published methods** (stationary bootstrap, FDR control)
- **Users might want this** for publication-quality results
- **Just expensive**, not useless (4000 bootstrap iterations)

**Recommendation: KEEP BUT:**
1. Make it easier to enable
2. Add progress bar for long computation
3. Document when to use it
4. Consider faster bootstrap methods

### 6. Single-Use Methods - KEEP FOR READABILITY ✅

**Methods in question:**
```python
_compute_stability()      # 29 lines - Clear purpose
_compute_dr_quality()      # 28 lines - Clear purpose  
_compute_robust_inference() # 38 lines - Clear purpose
_is_dr_estimator()        # 5 lines - Could inline
_has_influence_functions() # 11 lines - Could inline
```

**Analysis:**
- Large methods (`_compute_*`) improve readability
- Each handles a specific diagnostic domain
- Inlining would make `run()` method 150+ lines
- Small utilities could be inlined but minimal benefit

**Recommendation: KEEP AS-IS**
- Methods provide logical separation
- Easier to test individually
- Easier to maintain
- Only inline the trivial ones if doing other changes

## Implementation Plan

### Phase 1: Delete Gates (Immediate)

```bash
# 1. Remove gates.py
rm cje/utils/diagnostics/gates.py

# 2. Remove from __init__.py
# Remove: GateStatus, GateResult, GateReport, run_diagnostic_gates

# 3. Remove from DiagnosticSuite
# Remove: gate_report field
# Remove: gate-related methods

# 4. Remove from DiagnosticRunner
# Remove: _run_gates, _suite_to_gate_format methods
# Remove: run_gates, gate_config from DiagnosticConfig

# 5. Remove from base_estimator
# Remove: run_gates, gate_config parameters
# Remove: _run_diagnostic_gates method

# 6. Remove from tests
rm cje/tests/test_gates.py
```

**Impact: -785 lines, zero user impact**

### Phase 2: Remove Useless Fields

```python
# In DiagnosticSuite, remove:
- computation_time: Optional[float] = None
- estimator_type: str = "unknown"

# In DiagnosticRunner.run(), remove:
- suite.computation_time = time.time() - start_time
- estimator_type=estimator.__class__.__name__,

# In display.py, remove:
- if suite.computation_time: ...
- lines.append(f"Estimator Type: {suite.estimator_type}")
```

**Impact: -30 lines, cleaner data structure**

### Phase 3: Improve Robust Inference (Future)

```python
# Make it easier to enable:
class DiagnosticConfig:
    compute_robust_se: bool = False  # Keep off by default
    # But add:
    quick_robust_se: bool = False  # Faster method with 1000 iterations
    
# Add progress indication:
if self.config.compute_robust_se:
    print("Computing robust inference (this may take a minute)...")
    with tqdm(total=self.config.n_bootstrap) as pbar:
        suite.robust_inference = self._compute_robust_inference(result, pbar)
```

**Impact: Better UX, same functionality**

## What We're NOT Removing

### 1. Stability Diagnostics ✅
- **Valuable**: Detects distribution drift
- **Enabled by default**: Actually runs
- **Lightweight**: Fast computation
- **Actionable**: Warns about problems

### 2. Weight Diagnostics ✅
- **Core functionality**: Essential for IPS methods
- **Always needed**: Can't do IPS without weights
- **Well-designed**: Clean, useful metrics

### 3. DR Quality Metrics ✅
- **Method-specific**: Only for DR estimators
- **Valuable**: Orthogonality is important
- **Published**: Based on paper's approach

## Final Recommendations

### DO NOW (Safe, High Value)
1. **Delete gates.py entirely** - 785 lines of unused code
2. **Remove gate_report field** - Never used
3. **Remove computation_time** - Not useful
4. **Remove estimator_type** - Redundant

**Total removal: ~815 lines**

### DON'T DO (Has Value)
1. **Keep robust_inference** - Real statistical value
2. **Keep single-use methods** - Improve readability
3. **Keep stability diagnostics** - Valuable and used
4. **Keep weight diagnostics** - Essential

### MAYBE LATER (Needs Thought)
1. **Simplify robust_inference API** - Make easier to use
2. **Add quick_robust mode** - Faster alternative
3. **Progressive computation** - Show results as computed

## Risk Assessment

**Removing gates.py:**
- Risk: **ZERO** - Never enabled, no users
- Benefit: **HIGH** - 785 lines removed

**Removing fields:**
- Risk: **MINIMAL** - Not used in critical paths  
- Benefit: **MEDIUM** - Cleaner structure

**Keeping robust_inference:**
- Risk: **NONE** - Already there, just disabled
- Benefit: **FUTURE** - Users might want this

## The Philosophy

**Delete with confidence when:**
- Never enabled by default (gates)
- No clear user value (computation_time)
- Redundant information (estimator_type)
- Over-engineered for the problem (gates)

**Keep with purpose when:**
- Has statistical/mathematical value (robust_inference)
- Improves code organization (separate methods)
- Users might reasonably want it (robust CI)
- Based on published methods (bootstrap, FDR)

## Bottom Line

Gates.py is **pure technical debt** - complex code that provides no value. Delete it.

The other components are **dormant features** - they have potential value but need better exposure.

Focus on removing the debt (gates), not the features (robust inference).