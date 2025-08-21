# MRDR Fold System Integration Plan

## Current State
MRDR currently has a hybrid approach to fold management:
1. **Primary**: Tries to read `cv_fold` from metadata (lines 194-200)
2. **Fallback**: Creates its own KFold if no cv_fold found (lines 246-252)
3. **Issue**: Still relies on metadata storage which we're trying to eliminate

## Proposed Changes

### 1. Update MRDREstimator.fit()
Replace the cv_fold metadata reading with direct fold computation:

```python
# OLD (lines 193-200)
cv_map = {
    str(s.prompt_id): int(s.metadata["cv_fold"])
    for s in calibrated_dataset.samples
    if "cv_fold" in s.metadata and s.metadata["cv_fold"] is not None
}

# NEW
from ..data.folds import get_fold
cv_map = {
    str(s.prompt_id): get_fold(s.prompt_id, self.n_folds, self.random_seed)
    for s in calibrated_dataset.samples
}
```

### 2. Remove KFold Fallback
Since we'll always have prompt_ids, we can remove the KFold fallback:

```python
# DELETE lines 246-252 (KFold creation)
# Always use unified fold system
```

### 3. Update Fresh Draw Fold Assignment
Currently assigns all fresh draws the same fold (line 371). Should compute per prompt:

```python
# OLD
fresh_fold_ids = np.full(len(fresh_scores), fold_ids[i])

# NEW
from ..data.folds import get_fold
fresh_fold_ids = np.array([
    get_fold(pid, self.n_folds, self.random_seed) 
    for pid in fresh_prompt_ids
])
```

### 4. Simplify Fold ID Collection
The complex logic for collecting fold_ids can be simplified:

```python
# Always compute from prompt_ids
fold_ids = np.array([
    get_fold(pid, self.n_folds, self.random_seed)
    for pid in prompt_ids
])
```

## Benefits
1. **Consistency**: MRDR uses same fold assignment as all other components
2. **Simplicity**: No more checking metadata or fallback logic
3. **Robustness**: Works with filtered data and fresh draws
4. **Maintainability**: Single source of truth for fold assignment

## Implementation Steps
1. Import `get_fold` from `cje.data.folds`
2. Replace cv_fold metadata reading with direct computation
3. Remove KFold fallback logic
4. Update fresh draw fold assignment
5. Test with existing MRDR tests

## Testing Approach
```python
# Verify MRDR gets same folds as other estimators
from cje.data.folds import get_fold

# For each sample in dataset
for sample in dataset.samples:
    mrdr_fold = get_fold(sample.prompt_id, n_folds=5, seed=42)
    dr_fold = get_fold(sample.prompt_id, n_folds=5, seed=42)
    assert mrdr_fold == dr_fold
```

## Risk Assessment
- **Low Risk**: Changes are localized to MRDR
- **Backward Compatible**: Will work even if cv_fold still in metadata
- **Performance**: O(1) hashing is fast

## Estimated Time
- Implementation: 30 minutes
- Testing: 30 minutes
- Total: 1 hour