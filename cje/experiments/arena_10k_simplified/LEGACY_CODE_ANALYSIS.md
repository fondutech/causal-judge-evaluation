# Legacy Code Analysis for Arena 10K Experiment

## 🔍 Deep Dive Findings

### 1. **migrate_prompt_id_if_needed() in analyze_dataset.py** ⚠️
- **Status**: LEGACY - Can be removed
- **Purpose**: Migrates prompt_id from metadata to top-level
- **Current Reality**: All our data already has prompt_id at top-level
- **Evidence**: `has_prompt_id: true, has_metadata_prompt_id: false`
- **Recommendation**: DELETE this function and its call on line 762

### 2. **oracle_diagnostics.py** ❌
- **Status**: UNUSED
- **Lines**: 331
- **Usage Count**: 0 references in other files
- **Purpose**: Advanced R²(S→Y) diagnostics
- **Recommendation**: MOVE to a separate research/experiments folder or DELETE

### 3. **create_oracle_coverage_variants.py** ⚠️
- **Status**: RESEARCH TOOL
- **Lines**: 261
- **Usage Count**: Only self-reference
- **Purpose**: Creates dataset variants for ablation studies
- **Recommendation**: MOVE to research folder or document as optional

### 4. **Hardcoded n_folds=5** 📊
- **Files**: analyze_dataset.py, reward_utils.py, analyze_oracle_coverage.py
- **Pattern**: `n_folds=5` appears 8+ times
- **Issue**: Should come from experiment_config.py
- **Recommendation**: Add to experiment_config.py

### 5. **Comments Referencing "Old Codebase"** 📝
- **File**: pipeline_steps/prepare_arena_data.py
- Lines 6 & 116: "key insight from the old codebase"
- **Issue**: These comments are confusing - what old codebase?
- **Recommendation**: REMOVE or clarify these comments

### 6. **Threshold Magic Numbers** 🔢
- **analyze_dataset.py**:
  - `--extreme-threshold-high` default=100.0
  - `--extreme-threshold-low` default=0.01
- **prepare_arena_data.py**:
  - Line 39: "Threshold of 0.3 filters ~0.28% of prompts"
- **Issue**: Magic numbers without config
- **Recommendation**: Move to experiment_config.py

## 📋 Action Items

### High Priority (Remove/Fix Now)
1. ❌ **DELETE** `migrate_prompt_id_if_needed()` function (lines 91-140)
2. ❌ **DELETE** the call to it on line 762
3. ❌ **DELETE** or **MOVE** `oracle_diagnostics.py` (unused)
4. ✏️ **UPDATE** comments about "old codebase" in prepare_arena_data.py

### Medium Priority (Consolidate)
5. 📦 **ADD** `N_FOLDS = 5` to experiment_config.py
6. 📦 **ADD** threshold constants to experiment_config.py
7. 🚚 **MOVE** research scripts to `research/` subfolder:
   - `oracle_diagnostics.py`
   - `create_oracle_coverage_variants.py`
   - `analyze_oracle_coverage.py`

### Low Priority (Documentation)
8. 📝 **DOCUMENT** which scripts are core vs research in README

## 🎯 Code Quality Issues

### Duplicate Batch Processing Logic
- `evaluation_utils.py` has `score_batch()`
- `add_scores_with_resume.py` reimplements similar logic
- **Not Critical**: They serve different purposes (one is generic, one has resume capability)

### Inconsistent Error Handling
- Some scripts use `skip_failures=True`
- Others raise immediately
- **Recommendation**: Standardize based on use case

## 💡 Overall Assessment

The codebase is generally clean but has accumulated some legacy cruft:
- **15% legacy/unused code** (migrate function, oracle_diagnostics)
- **10% research code** mixed with production
- **75% clean, necessary code**

After cleanup, we'd have:
- ~4,500 lines of production code (from ~5,200)
- Clear separation of research vs production
- All configuration centralized
- No legacy migration code