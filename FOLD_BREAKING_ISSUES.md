# Critical Issues Found with Fold Management Changes

## Summary
Our unified fold management implementation introduced several **critical bugs** that break DR estimation and oracle slice augmentation. While the core fold system works, the integration points were not properly updated.

## Critical Issues Found

### 1. **DR Estimator Fold Indexing Bug** 🔴 CRITICAL
**Location**: `cje/estimators/dr_base.py:273-274`

**Problem**: 
- `self.fold_assignments` is indexed by position in FULL dataset (100 samples)
- But `idx` at line 274 refers to position in FILTERED valid samples
- This causes **wrong fold assignments** for filtered samples

**Example**:
```python
# self.fold_assignments has 100 entries for full dataset
# But when we filter to 80 valid samples:
self.fold_assignments[idx]  # idx is 0-79, but refers to filtered position!
# This gets the wrong fold!
```

**Impact**: Cross-fitting is completely broken for DR when samples are filtered

### 2. **MRDR Still Reads cv_fold from Metadata** 🔴 CRITICAL
**Location**: `cje/estimators/mrdr.py:194-196`

**Problem**:
```python
cv_map = {
    str(s.prompt_id): int(s.metadata["cv_fold"])  # cv_fold no longer exists!
    for s in self.sampler.dataset.samples
    if "cv_fold" in s.metadata and s.metadata["cv_fold"] is not None
}
```

**Impact**: 
- MRDR will never find cv_fold in metadata
- Falls back to creating its own KFold
- **Inconsistent folds between calibration and MRDR**

### 3. **CalibratedIPS Can't Extract Fold IDs** 🟡 MAJOR
**Location**: `cje/estimators/calibrated_ips.py:158`

**Problem**:
```python
fold_list = [d.get("cv_fold") for d in data]  # cv_fold not in data dict anymore
```

**Impact**: 
- Oracle augmentation won't use cross-fitting
- Falls back to global fitting
- **Reduced statistical efficiency**

### 4. **Oracle Slice Augmentation Missing Folds** 🟡 MAJOR
**Location**: `cje/calibration/oracle_slice.py:78-101`

**Problem**:
- Expects cv_folds parameter to enable cross-fitting
- But we no longer pass cv_fold through the pipeline
- Falls back to non-cross-fitted version

**Impact**: 
- Less accurate m̂(S) estimation
- **Potentially optimistic confidence intervals**

### 5. **PrecomputedSampler Doesn't Include cv_fold** 🟡 MAJOR
**Location**: `cje/data/precomputed_sampler.py:191`

**Problem**:
We removed cv_fold from the data dictionary:
```python
# Note: cv_fold no longer stored - computed on-demand from prompt_id
```

**Impact**: 
- All downstream components expecting cv_fold in data fail
- Can't pass fold information through data pipeline

## Root Cause Analysis

The fundamental issue is a **leaky abstraction**:
1. We removed cv_fold from metadata storage (good)
2. But many components still expect to receive fold information through data dictionaries
3. We didn't provide an alternative way to get folds for FILTERED data

## Why Tests Didn't Catch This

1. **Unit tests mock data**: Don't test real data flow
2. **No integration tests**: For the full calibration → sampler → estimator pipeline
3. **Test datasets are complete**: No filtering happens, so indexing bug doesn't show
4. **Missing fold consistency tests**: No tests verify same folds across components

## Proposed Fixes

### Fix 1: DR Estimator Fold Indexing
```python
# Instead of self.fold_assignments[idx] where idx is filtered position
# Use prompt_id to get correct fold:
from ..data.folds import get_fold
fold = get_fold(sample.prompt_id, self.n_folds, self.random_seed)
```

### Fix 2: PrecomputedSampler get_folds_for_policy()
Already implemented but not used! Need to update estimators to call:
```python
folds = sampler.get_folds_for_policy(policy, n_folds, seed)
```

### Fix 3: Pass Folds Through Pipeline
Option A: Re-add cv_fold to data dict (but compute on-demand)
Option B: Pass fold array separately through the pipeline
Option C: Have each component compute folds from prompt_ids

### Fix 4: MRDR Fold Consistency
```python
# Instead of reading cv_fold from metadata
# Use unified fold system:
from cje.data.folds import get_fold
fold = get_fold(sample.prompt_id, n_folds, seed)
```

## Impact Assessment

| Component | Severity | Current State | User Impact |
|-----------|----------|--------------|-------------|
| DR Estimator | 🔴 Critical | Wrong folds for filtered data | Incorrect estimates |
| MRDR | 🔴 Critical | Creates own folds | Inconsistent cross-fitting |
| TMLE | 🔴 Critical | Same as DR | Incorrect estimates |
| CalibratedIPS | 🟡 Major | No cross-fitted augmentation | Less efficient |
| Oracle Augmentation | 🟡 Major | Falls back to global | Optimistic CIs |

## Immediate Actions Needed

1. **REVERT or FIX URGENTLY** - DR estimation is broken
2. Add integration tests for full pipeline
3. Fix fold indexing in DR estimator
4. Update all components to use unified fold system properly
5. Add fold consistency checks

## Lessons Learned

1. **Removing stored state requires providing alternatives**
   - We removed cv_fold storage but didn't provide ways to get folds for filtered data

2. **Integration points are critical**
   - Unit tests aren't enough - need full pipeline tests

3. **Backward compatibility isn't just about APIs**
   - Data flow compatibility matters too

4. **Index-based systems are fragile**
   - Filtering breaks index assumptions

5. **Test with realistic data**
   - Need tests with filtered samples, missing data, etc.

## Recommendation

**We should either:**
1. **Fix forward**: Implement all fixes listed above (2-3 hours)
2. **Partial revert**: Keep unified folds but restore cv_fold in data dict
3. **Full revert**: Go back to original system until fixes ready

Given the critical nature of these bugs, I recommend **partial revert** - keep the unified fold system but restore cv_fold in the data dictionary so downstream components work correctly.