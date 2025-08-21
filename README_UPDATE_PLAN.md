# README Update Plan for Unified Fold Management

## Files Requiring Updates

### 1. `cje/data/README.md` ⚠️ HIGH PRIORITY
**Current Issues**:
- Line 124: States `cv_fold` is in metadata (NO LONGER STORED)
- Line 141: Shows example with `cv_fold: 2` in metadata
- Missing documentation of new `folds.py` module

**Required Changes**:
```markdown
# Remove from metadata description (line 124)
- Remove: "cv_fold: Cross-validation fold assignment"

# Update example (lines 128-143)
- Remove "cv_fold": 2 from metadata example

# Add new section about fold management
## Fold Management

The `folds` module provides unified cross-validation fold assignment:

### Core Functions
- `get_fold(prompt_id, n_folds=5, seed=42)`: Get fold for single prompt
- `get_folds_for_prompts(prompt_ids, ...)`: Vectorized fold assignment
- `get_folds_for_dataset(dataset, ...)`: Dataset-level folds
- `get_folds_with_oracle_balance(...)`: Balanced oracle distribution

### Key Properties
- Deterministic: `hash(prompt_id) % n_folds`
- Filtering-proof: Based on stable IDs, not indices
- Fresh-draw compatible: Same prompt_id → same fold

Note: Folds are computed on-demand, not stored in metadata.
```

### 2. `cje/calibration/README.md` ✅ MOSTLY ACCURATE
**Current State**: 
- Correctly describes cross-fitting concepts
- Mentions fold tracking through pipeline

**Minor Updates**:
- Add note that JudgeCalibrator now uses unified fold system
- Clarify that fold_ids come from `data.folds` module

### 3. `cje/estimators/README.md` ✅ MOSTLY ACCURATE
**Current State**:
- Correctly describes cross-fitting for DR methods
- General concepts remain valid

**Minor Updates**:
- Add note about unified fold management
- Mention that all estimators use same fold system

### 4. `cje/interface/README.md` ✅ OK
**Current State**:
- Shows n_folds configuration option
- No incorrect information

**Optional Enhancement**:
- Could add note that n_folds affects all components consistently

### 5. Main `README.md` ✅ OK
**Current State**:
- High-level overview, no fold details
- Pipeline diagram remains accurate

**No Changes Needed**

### 6. `cje/experiments/arena_10k_simplified/README.md` ❓ CHECK
**Need to Review**:
- May reference cv_fold storage
- May describe old fold system

### 7. Other Module READMEs ✅ OK
- `diagnostics/README.md` - No fold mentions
- `teacher_forcing/README.md` - No fold mentions
- `utils/README.md` - No fold mentions
- `visualization/README.md` - No fold mentions
- `tests/README.md` - No fold mentions

## Priority Order

1. **CRITICAL**: Update `cje/data/README.md`
   - Remove cv_fold from metadata docs
   - Add folds.py documentation
   - Update examples

2. **NICE TO HAVE**: Minor updates to calibration/estimators READMEs
   - Add notes about unified system
   - Clarify fold source

3. **OPTIONAL**: Check experiments README
   - Verify no outdated information

## Implementation Script

```bash
# 1. Update data/README.md
# Remove cv_fold references
# Add folds.py documentation

# 2. Add note to calibration/README.md about unified folds

# 3. Add note to estimators/README.md about consistent folds

# 4. Commit all README updates
git add -A
git commit -m "docs: Update READMEs for unified fold management

- Remove cv_fold from data/README.md metadata description
- Document new folds.py module and its functions
- Update example to remove cv_fold from metadata
- Add notes about unified fold system to relevant READMEs"
```

## Key Messages to Convey

1. **Folds are computed, not stored**
   - Dynamic calculation via `hash(prompt_id) % n_folds`
   - No cv_fold in metadata anymore

2. **Single source of truth**
   - All components use `cje.data.folds`
   - Consistent across entire pipeline

3. **Benefits**
   - Deterministic and reproducible
   - Survives filtering
   - Works with fresh draws

4. **Migration**
   - Components handle missing cv_fold gracefully
   - Backward compatible where possible