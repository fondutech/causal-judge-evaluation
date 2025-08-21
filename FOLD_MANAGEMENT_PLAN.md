# Unified Fold Management Implementation Plan

## Executive Summary

CJE currently has **five independent fold assignment systems** that create inconsistent cross-validation folds across components. This causes:
- Same sample in different folds across estimators
- Broken fold alignment after filtering
- Fresh draws unable to inherit correct folds
- Corrupted orthogonality in DR methods

**Solution**: Create a single, authoritative fold assignment system based on prompt_id hashing.

## Current State Analysis

### Five Independent Systems

1. **JudgeCalibrator** (`calibration/judge.py`)
   - KFold on oracle samples for balance
   - Hash-based assignment for unlabeled samples
   - Stores in `_fold_ids`

2. **DREstimator** (`estimators/dr_base.py`)
   - Simple `index % n_folds` + shuffle
   - Creates via `_create_fold_assignments()`
   - Stores in `self.fold_assignments`

3. **StackedDREstimator** (`estimators/stacking.py`)
   - Independent KFold
   - Stores in `self.shared_fold_ids`

4. **SIMCal** (`calibration/simcal.py`)
   - Internal KFold for OOF weight stacking
   - Independent of data sample folds

5. **MRDR** (`estimators/mrdr.py`)
   - Another KFold for robust weight combination

### Critical Issues

#### Issue 1: Filtering Breaks Index-Based Folds
```python
# Original: 1000 samples with folds [0,1,2,3,4,0,1,2,...]
# After filtering: 800 samples
# DR creates NEW folds: [0,1,2,3,4,0,1,2,...] for 800 samples
# Complete misalignment!
```

#### Issue 2: Fresh Draws Can't Inherit Folds
```python
# Logged data: prompt_123 → fold 2 (via index-based assignment)
# Fresh draw: prompt_123 → fold ? (no way to know original fold)
```

#### Issue 3: Components Use Different Folds
```python
# Same sample:
# - JudgeCalibrator: fold 2
# - DREstimator: fold 0  
# - StackedDR: fold 4
# Breaks cross-fitting orthogonality!
```

## Design Decisions

### Where Should Fold Management Live?

#### Option A: `/data` directory ✓
**Pros:**
- Folds are fundamentally about organizing data
- `PrecomputedSampler` already lives here and needs fold access
- `Dataset` and `Sample` models are here
- Natural import: `from cje.data import get_folds`
- Aligns with "data transformation" mental model

**Cons:**
- Most usage is in calibration module
- Cross-validation is more algorithmic than data

#### Option B: `/calibration` directory
**Pros:**
- 80% of fold usage is in calibration
- Cross-validation is fundamentally about fitting/calibration
- `JudgeCalibrator.fit_cv()` is the main CV entry point
- Keeps CV logic cohesive

**Cons:**
- Folds are used beyond just calibration
- Creates coupling between data and calibration modules

#### Option C: `/utils` directory
**Pros:**
- Neutral location
- Already has utility functions

**Cons:**
- Utils is for export/analysis, not core functionality
- Less discoverable

**Recommendation: `/data` directory**

Reasoning:
1. Folds are a property of the data, not the algorithm
2. `PrecomputedSampler` (in `/data`) is the natural place to expose folds
3. Cleaner dependency graph: calibration depends on data, not vice versa
4. More intuitive: "get folds for this dataset"

### Architecture: Simple Functions vs Classes

**Decision: Simple functions**

```python
# YES: Simple, clear, testable
fold = get_fold(prompt_id)

# NO: Overengineered
fold = FoldAssigner().get_fold(prompt_id)
```

Reasoning:
- CJE pattern: Most utilities are functions
- No state needed: Folds are pure functions of prompt_id
- Easier testing: Pure functions are trivial to test
- Clear API: Can't get simpler than `get_fold(prompt_id)`

## Implementation Design

### Core Module: `cje/data/folds.py`

```python
"""Unified fold assignment for cross-validation.

Core principle: Use prompt_id hashing for stable fold assignment
that survives filtering and works across all components.

All cross-validation in CJE MUST use these functions.
"""

import hashlib
import numpy as np
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Dataset

def get_fold(prompt_id: str, n_folds: int = 5, seed: int = 42) -> int:
    """Get fold assignment for a single prompt_id.
    
    This is THE authoritative way to assign folds in CJE.
    Uses stable hashing that:
    - Survives sample filtering
    - Works with fresh draws (same prompt_id → same fold)
    - Ensures consistency across all components
    
    Args:
        prompt_id: Unique identifier for the prompt
        n_folds: Number of folds for cross-validation
        seed: Random seed for reproducibility
        
    Returns:
        Fold index in [0, n_folds)
        
    Example:
        >>> get_fold("prompt_123")  # Always returns same fold
        2
        >>> get_fold("prompt_123", n_folds=10)  # Different for different n_folds
        7
    """
    hash_input = f"{prompt_id}-{seed}-{n_folds}".encode()
    hash_bytes = hashlib.blake2b(hash_input, digest_size=8).digest()
    return int.from_bytes(hash_bytes, 'big') % n_folds

def get_folds_for_prompts(
    prompt_ids: List[str], 
    n_folds: int = 5, 
    seed: int = 42
) -> np.ndarray:
    """Get fold assignments for multiple prompt_ids.
    
    Args:
        prompt_ids: List of prompt identifiers
        n_folds: Number of folds
        seed: Random seed
        
    Returns:
        Array of fold indices, shape (len(prompt_ids),)
    """
    return np.array([get_fold(pid, n_folds, seed) for pid in prompt_ids])

def get_folds_for_dataset(
    dataset: 'Dataset',
    n_folds: int = 5,
    seed: int = 42
) -> np.ndarray:
    """Get fold assignments for all samples in a dataset.
    
    Args:
        dataset: Dataset with samples containing prompt_ids
        n_folds: Number of folds
        seed: Random seed
        
    Returns:
        Array of fold indices aligned with dataset.samples
    """
    prompt_ids = [s.prompt_id for s in dataset.samples]
    return get_folds_for_prompts(prompt_ids, n_folds, seed)

# Migration helper for JudgeCalibrator
def get_folds_with_oracle_balance(
    prompt_ids: List[str],
    oracle_mask: np.ndarray,
    n_folds: int = 5,
    seed: int = 42
) -> np.ndarray:
    """Get folds with balanced oracle sample distribution.
    
    Ensures oracle samples are evenly distributed across folds
    (important for small oracle subsets). Unlabeled samples
    use standard hash-based assignment.
    
    Args:
        prompt_ids: All prompt identifiers
        oracle_mask: Boolean mask indicating oracle samples
        n_folds: Number of folds
        seed: Random seed
        
    Returns:
        Array of fold indices with balanced oracle distribution
        
    Note:
        This is primarily for JudgeCalibrator backward compatibility.
        New code should use get_folds_for_prompts() directly.
    """
    n = len(prompt_ids)
    folds = np.zeros(n, dtype=int)
    
    # Oracle samples: round-robin for perfect balance
    oracle_indices = np.where(oracle_mask)[0]
    if len(oracle_indices) > 0:
        # Shuffle oracle indices for randomization
        rng = np.random.RandomState(seed)
        oracle_indices = oracle_indices.copy()
        rng.shuffle(oracle_indices)
        
        for i, idx in enumerate(oracle_indices):
            folds[idx] = i % n_folds
    
    # Unlabeled samples: standard hash-based
    unlabeled = ~oracle_mask
    if np.any(unlabeled):
        unlabeled_ids = [prompt_ids[i] for i in range(n) if unlabeled[i]]
        unlabeled_folds = get_folds_for_prompts(unlabeled_ids, n_folds, seed)
        folds[unlabeled] = unlabeled_folds
    
    return folds
```

### Update `data/__init__.py`

```python
# Add to exports
from .folds import (
    get_fold,
    get_folds_for_prompts,
    get_folds_for_dataset,
    get_folds_with_oracle_balance,
)

__all__ = [
    # ... existing exports ...
    # Fold management
    "get_fold",
    "get_folds_for_prompts", 
    "get_folds_for_dataset",
    "get_folds_with_oracle_balance",
]
```

## Component Updates

### 1. PrecomputedSampler Enhancement

```python
# data/precomputed_sampler.py
from .folds import get_folds_for_prompts

class PrecomputedSampler:
    def get_folds_for_policy(
        self, 
        policy: str,
        n_folds: int = 5,
        seed: int = 42
    ) -> Optional[np.ndarray]:
        """Get consistent fold assignments for policy's valid samples.
        
        Returns folds for the FILTERED samples that align with
        get_data_for_policy(). This ensures folds match the actual
        data used for estimation.
        
        Args:
            policy: Target policy name
            n_folds: Number of cross-validation folds
            seed: Random seed for reproducibility
            
        Returns:
            Fold assignments for valid samples, or None if no data
        """
        data = self.get_data_for_policy(policy)
        if data is None:
            return None
        
        prompt_ids = [d["prompt_id"] for d in data]
        return get_folds_for_prompts(prompt_ids, n_folds, seed)
```

### 2. JudgeCalibrator Simplification

```python
# calibration/judge.py
from ..data.folds import get_folds_for_prompts, get_folds_with_oracle_balance

class JudgeCalibrator:
    def __init__(self, random_seed: int = 42, balance_oracle_folds: bool = True):
        self.random_seed = random_seed
        self.balance_oracle_folds = balance_oracle_folds  # New option
        
    def fit_cv(self, judge_scores, oracle_labels, oracle_mask, n_folds):
        # Get prompt_ids from dataset (need to pass or store)
        prompt_ids = [s.prompt_id for s in self.dataset.samples]
        
        # REPLACE complex KFold + hash mixture with:
        if self.balance_oracle_folds:
            # Ensure oracle samples evenly distributed
            self._fold_ids = get_folds_with_oracle_balance(
                prompt_ids, oracle_mask, n_folds, self.random_seed
            )
        else:
            # Simple hash-based for all samples
            self._fold_ids = get_folds_for_prompts(
                prompt_ids, n_folds, self.random_seed
            )
        
        # Rest of the method unchanged...
```

### 3. DREstimator Cleanup

```python
# estimators/dr_base.py
from ..data.folds import get_folds_for_dataset

class DREstimator(BaseCJEEstimator):
    def __init__(self, ...):
        # DELETE _create_fold_assignments method entirely
        
    def fit(self):
        # REPLACE fold creation with:
        self.fold_assignments = get_folds_for_dataset(
            self.sampler.dataset, 
            self.n_folds, 
            self.random_seed
        )
        
        # Rest unchanged...
```

### 4. StackedDREstimator Alignment

```python
# estimators/stacking.py
from ..data.folds import get_folds_for_dataset

class StackedDREstimator:
    def setup_shared_resources(self):
        # DELETE KFold creation
        # REPLACE with:
        self.shared_fold_ids = get_folds_for_dataset(
            self.sampler.dataset, 
            5,  # or self.n_folds if configurable
            self.seed
        )
```

### 5. Remove cv_fold from Metadata

Since folds are computed deterministically from prompt_id, no need to store:

```python
# calibration/dataset.py
def calibrate_dataset(...):
    # DELETE these lines:
    # if enable_cross_fit and result.fold_ids is not None:
    #     new_metadata["cv_fold"] = int(result.fold_ids[i])
    
    # Folds are computed on-demand from prompt_id
```

## Test Strategy

### Core Tests: `tests/test_unified_folds.py`

```python
import numpy as np
import pytest
from cje.data import (
    get_fold, 
    get_folds_for_dataset,
    get_folds_for_prompts,
    get_folds_with_oracle_balance
)
from cje.data.models import Dataset, Sample

def test_fold_determinism():
    """Same prompt_id always gets same fold."""
    fold1 = get_fold("test_123")
    fold2 = get_fold("test_123")
    assert fold1 == fold2
    
    # Different seeds give different folds
    fold3 = get_fold("test_123", seed=99)
    assert fold3 != fold1  # Usually different

def test_fold_range():
    """Folds are in correct range."""
    for n_folds in [2, 5, 10]:
        fold = get_fold("test", n_folds=n_folds)
        assert 0 <= fold < n_folds

def test_fold_survives_filtering():
    """Folds unchanged after filtering samples."""
    # Create dataset
    samples = []
    for i in range(100):
        samples.append(Sample(
            prompt_id=f"prompt_{i}",
            prompt=f"Test prompt {i}",
            response=f"Response {i}",
            reward=np.random.random(),
            target_policy_logprobs={"policy": -10.0}
        ))
    dataset = Dataset(samples=samples, target_policies=["policy"])
    
    # Get folds before filtering
    folds_before = get_folds_for_dataset(dataset)
    
    # Filter to subset
    filtered_samples = [s for s in samples if s.reward > 0.5]
    filtered_dataset = Dataset(samples=filtered_samples, target_policies=["policy"])
    folds_after = get_folds_for_dataset(filtered_dataset)
    
    # Check consistency
    for i, sample in enumerate(filtered_dataset.samples):
        # Find original index
        orig_idx = next(
            j for j, s in enumerate(dataset.samples) 
            if s.prompt_id == sample.prompt_id
        )
        assert folds_after[i] == folds_before[orig_idx]

def test_fresh_draws_inherit_folds():
    """Fresh draws with same prompt_id get same fold."""
    prompt_id = "shared_prompt_123"
    
    # Logged data fold
    logged_fold = get_fold(prompt_id)
    
    # Fresh draw with same prompt_id
    fresh_fold = get_fold(prompt_id)
    
    assert logged_fold == fresh_fold

def test_oracle_balance():
    """Oracle samples are balanced across folds."""
    n = 100
    prompt_ids = [f"p_{i}" for i in range(n)]
    
    # 20 oracle samples
    oracle_mask = np.zeros(n, dtype=bool)
    oracle_mask[:20] = True
    
    folds = get_folds_with_oracle_balance(
        prompt_ids, oracle_mask, n_folds=5
    )
    
    # Check oracle distribution
    oracle_folds = folds[oracle_mask]
    for fold in range(5):
        count = np.sum(oracle_folds == fold)
        assert count == 4  # 20 oracle / 5 folds = 4 per fold

def test_all_components_same_folds():
    """Verify all estimators would get identical folds."""
    dataset = create_test_dataset()
    
    # Simulate what each component would compute
    judge_folds = get_folds_for_dataset(dataset)
    dr_folds = get_folds_for_dataset(dataset)  
    stacked_folds = get_folds_for_dataset(dataset)
    
    # All should be identical
    np.testing.assert_array_equal(judge_folds, dr_folds)
    np.testing.assert_array_equal(dr_folds, stacked_folds)

def test_performance():
    """Fold assignment should be fast even for large datasets."""
    import time
    
    n = 100000
    prompt_ids = [f"prompt_{i}" for i in range(n)]
    
    start = time.time()
    folds = get_folds_for_prompts(prompt_ids)
    elapsed = time.time() - start
    
    assert len(folds) == n
    assert elapsed < 1.0  # Should take < 1 second for 100k samples
```

## Implementation Timeline

### Day 1: Core Infrastructure (4 hours)
- [ ] Create `cje/data/folds.py` with all functions (1 hour)
- [ ] Write comprehensive tests in `test_unified_folds.py` (2 hours)
- [ ] Update `data/__init__.py` exports (15 min)
- [ ] Run tests, fix any issues (45 min)

### Day 2: Component Updates (4 hours)
- [ ] Update PrecomputedSampler with `get_folds_for_policy()` (30 min)
- [ ] Update JudgeCalibrator to use new system (1 hour)
- [ ] Update DREstimator and remove old method (1 hour)
- [ ] Update StackedDREstimator (30 min)
- [ ] Update MRDR if needed (30 min)
- [ ] Fix any test failures (30 min)

### Day 3: Cleanup and Validation (4 hours)
- [ ] Remove cv_fold from metadata storage (1 hour)
- [ ] Run full test suite (1 hour)
- [ ] Update any documentation (1 hour)
- [ ] Final validation with ablation experiments (1 hour)

## Benefits

1. **Single Source of Truth**: One implementation, consistent everywhere
2. **Filtering-Proof**: prompt_id hashing survives all data transformations
3. **Fresh Draw Compatible**: Automatic fold inheritance via prompt_id
4. **Simple API**: `get_fold(prompt_id)` - can't get simpler
5. **Deterministic**: Same inputs always give same outputs
6. **Fast**: O(1) fold assignment per sample
7. **Testable**: Pure functions with no hidden state

## Migration Impact

| Component | Lines Changed | Risk | Benefit |
|-----------|--------------|------|---------|
| data/folds.py (new) | +150 | None | Foundation for consistency |
| data/__init__.py | +10 | None | Clean exports |
| PrecomputedSampler | +20 | Low | New capability |
| JudgeCalibrator | -30 | Low | Simpler code |
| DREstimator | -20 | Low | Removes complexity |
| StackedDR | -10 | Low | Alignment |
| Tests | +200 | None | Validation |

**Total: ~300 lines changed, net reduction in complexity**

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Oracle imbalance with pure hashing | `get_folds_with_oracle_balance()` helper |
| Different components need different n_folds | Each can specify, but encourage consistency |
| Missing prompt_ids in old data | Generate from prompt text hash |
| Performance with large datasets | Hashing is O(1), can add caching if needed |

## Alternatives Considered

### Alternative 1: Store folds in Dataset
- ❌ Breaks when filtering
- ❌ Can't handle fresh draws
- ❌ Requires state management

### Alternative 2: Index-based with mapping
- ❌ Complex to maintain through operations
- ❌ Still breaks with fresh draws
- ❌ Requires careful bookkeeping

### Alternative 3: Random assignment
- ❌ Not reproducible
- ❌ Different across runs
- ❌ Can't debug issues

## Decision

**Implement prompt_id hashing in `/data/folds.py`**

This provides a clean, simple, and robust solution that fixes all current issues while being easy to understand and maintain. The implementation is straightforward and can be completed in 3 days.

## Next Steps

1. Review this plan with team
2. Create `cje/data/folds.py` 
3. Write tests first (TDD)
4. Update components incrementally
5. Validate with full test suite