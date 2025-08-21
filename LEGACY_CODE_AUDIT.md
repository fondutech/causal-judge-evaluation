# Legacy Code Audit: Fold Management

## Summary
Found 27 references to `cv_fold` across 11 files, plus 4 remaining `KFold` imports. Most are handled gracefully but some need attention.

## Critical Issues to Fix

### 1. ❌ `visualization/weight_dashboards.py` (Line 235)
```python
fold_list = [d.get("cv_fold") for d in data]
```
**Issue**: Tries to read cv_fold from data dict (won't exist)
**Impact**: Cross-fitted predictions will silently fail
**Fix**: Compute folds on-demand from prompt_id

### 2. ❌ `experiments/arena_10k_simplified/analysis/calibration.py` (Lines 114-116)
```python
if has_fold_ids:
    sample.metadata["cv_fold"] = int(cal_result.fold_ids[i])
```
**Issue**: Still writes cv_fold to metadata
**Impact**: Misleading - writes data that won't be used
**Fix**: Remove this code block entirely

### 3. ⚠️ `calibration/oracle_slice.py` (Multiple references)
```python
def fit_m_hat(cv_folds: Optional[np.ndarray] = None)
```
**Issue**: Expects cv_folds parameter
**Impact**: Works but uses outdated interface
**Fix**: Update to compute folds internally or accept prompt_ids

### 4. ⚠️ `tests/conftest.py` (Line ~90)
```python
"cv_fold": i % 5,  # 5-fold cross-validation
```
**Issue**: Test fixture creates cv_fold in metadata
**Impact**: Tests don't reflect production behavior
**Fix**: Remove cv_fold from test data

## KFold Imports Still Present

### 1. ✅ `calibration/judge.py` 
- Has fallback to KFold when no prompt_ids
- Working correctly with backward compatibility

### 2. ❌ `calibration/simcal.py`
```python
kf = KFold(n_splits=self.cfg.n_folds, shuffle=True, random_state=42)
```
**Issue**: Creates own KFold for weight stacking
**Impact**: Inconsistent folds with rest of system
**Fix**: Use unified fold system

### 3. ⚠️ `estimators/mrdr.py`
- Falls back to KFold when no cv_fold found
- Works but suboptimal

## Files with cv_fold References (Sorted by Priority)

### HIGH PRIORITY - Broken or Misleading
1. `visualization/weight_dashboards.py` - Reads cv_fold that doesn't exist
2. `experiments/arena_10k_simplified/analysis/calibration.py` - Writes cv_fold
3. `tests/conftest.py` - Creates incorrect test data
4. `tests/test_dr_diagnostics.py` - Creates cv_fold in test data

### MEDIUM PRIORITY - Working but Suboptimal
5. `calibration/oracle_slice.py` - Uses cv_folds parameter
6. `calibration/simcal.py` - Creates own KFold
7. `estimators/mrdr.py` - Tries to read cv_fold, falls back to KFold
8. `estimators/tmle.py` - Passes cv_folds to oracle augmentation
9. `estimators/calibrated_ips.py` - Tries to read cv_fold for visualization

### LOW PRIORITY - Comments or Handled
10. `calibration/dataset.py` - Just a comment noting removal
11. `data/precomputed_sampler.py` - Computes on-demand (correct!)
12. `estimators/dr_base.py` - Warning message about cv_fold
13. `estimators/outcome_models.py` - Warning message

## Recommended Actions

### Immediate Fixes (Breaking Issues)
```python
# 1. Fix visualization/weight_dashboards.py
# Replace: fold_list = [d.get("cv_fold") for d in data]
# With:
from cje.data.folds import get_fold
fold_list = [get_fold(d.get("prompt_id"), 5, 42) if d.get("prompt_id") else None for d in data]

# 2. Remove from calibration.py
# Delete lines that write cv_fold to metadata

# 3. Update test fixtures
# Remove "cv_fold" from metadata in conftest.py
```

### Medium-term Improvements
1. Update `oracle_slice.py` to use prompt_ids instead of cv_folds parameter
2. Update `simcal.py` to use unified fold system
3. Complete MRDR migration (per MRDR_FOLD_FIX_PLAN.md)

### Testing Updates
1. Remove cv_fold from all test fixtures
2. Add tests that verify cv_fold is NOT in metadata
3. Test that visualization works without cv_fold

## Code to Search and Replace

### Find all cv_fold reads:
```bash
grep -r "\.get.*cv_fold" --include="*.py"
grep -r "\[.*cv_fold.*\]" --include="*.py"
```

### Find all cv_fold writes:
```bash
grep -r "cv_fold.*=" --include="*.py"
grep -r "\"cv_fold\":" --include="*.py"
```

### Find all KFold usage:
```bash
grep -r "KFold(" --include="*.py"
```

## Impact Assessment

### What's Working
- ✅ PrecomputedSampler computes cv_fold on-demand (correct approach)
- ✅ DR estimators handle missing cv_fold gracefully
- ✅ JudgeCalibrator has proper fallback logic

### What's Broken
- ❌ Visualization silently fails to use cross-fitted predictions
- ❌ Test data doesn't reflect production (has cv_fold)
- ❌ Some analysis scripts still write cv_fold

### What's Suboptimal
- ⚠️ Multiple components still create their own folds
- ⚠️ Oracle augmentation uses outdated interface
- ⚠️ Tests create unrealistic data

## Migration Path

1. **Phase 1**: Fix breaking issues (visualization, test data)
2. **Phase 2**: Update oracle augmentation interface
3. **Phase 3**: Complete MRDR and SIMCal migrations
4. **Phase 4**: Remove all KFold imports except judge.py fallback
5. **Phase 5**: Add deprecation warnings for cv_fold parameters