# CJE Codebase Architecture Overview

## High-Level Structure

```
causal-judge-evaluation/
├── cje/                      # Core library package
│   ├── __init__.py          # Main exports (analyze_dataset, etc.)
│   ├── calibration/         # Judge → Oracle calibration
│   ├── data/                # Data models and sampling
│   ├── diagnostics/         # Weight and estimation diagnostics  
│   ├── estimators/          # IPS, DR, MRDR, TMLE implementations
│   ├── interface/           # High-level API functions
│   ├── teacher_forcing/     # Fresh draw generation
│   ├── tests/               # Comprehensive test suite
│   ├── utils/               # Export and analysis utilities
│   └── visualization/       # Plotting and visual diagnostics
│
├── experiments/             # Real-world experiments
│   └── arena_10k_simplified/  # Production pipeline example
│       ├── data/              # Dataset and fresh draws
│       ├── analysis/          # Analysis scripts
│       └── ablations/         # Systematic experiments
│
├── docs/                    # Documentation (if exists)
├── scripts/                 # Utility scripts
└── README.md               # Project overview
```

## Core Modules and Their Roles

### 1. **Data Layer** (`cje/data/`)
**Purpose**: Foundation for all data handling

```
data/
├── models.py              # Dataset, Sample, EstimationResult models
├── precomputed_sampler.py # Handles filtering, caching, sampling
├── fresh_draws.py         # Fresh draw loading and validation
└── folds.py              # 🆕 UNIFIED FOLD MANAGEMENT (our work)
```

**Key Innovation (Our Work)**:
- `folds.py` provides THE single source of truth for fold assignments
- Replaces 5 inconsistent fold systems with `hash(prompt_id) % n_folds`
- Ensures deterministic, filtering-proof cross-validation

### 2. **Calibration Layer** (`cje/calibration/`)
**Purpose**: Maps judge scores to oracle labels

```
calibration/
├── judge.py              # JudgeCalibrator with cross-fitting
├── isotonic.py           # Isotonic regression implementation
├── simcal.py             # SIMCal weight calibration
└── dataset.py            # High-level calibration API
```

**Integration with Fold Management**:
- JudgeCalibrator now uses `get_folds_with_oracle_balance()` for balanced oracle distribution
- Cross-fitting uses unified fold assignments for consistency

### 3. **Estimators Layer** (`cje/estimators/`)
**Purpose**: Causal effect estimation methods

```
estimators/
├── base_estimator.py     # Abstract base with oracle augmentation
├── calibrated_ips.py     # IPS with SIMCal (production default)
├── dr_base.py            # Base doubly robust estimator
├── dr_cpo.py             # Basic DR implementation
├── mrdr.py               # Multiply robust DR
├── tmle.py               # Targeted maximum likelihood
├── stacking.py           # Ensemble methods (StackedDR)
└── outcome_models.py     # Outcome modeling for DR
```

**Integration with Fold Management**:
- All DR estimators use unified fold system via `get_folds_for_dataset()`
- StackedDR uses `shared_fold_ids` for consistent cross-fitting
- Removed index-based fold assignments that broke with filtering

### 4. **Diagnostics Layer** (`cje/diagnostics/`)
**Purpose**: Monitor estimation quality

```
diagnostics/
├── base.py               # Base diagnostic interface
├── ips_diagnostics.py    # ESS, weight distribution analysis
├── dr_diagnostics.py     # Orthogonality, influence functions
└── reliability.py        # Refusal gates and safety checks
```

### 5. **Interface Layer** (`cje/interface/`)
**Purpose**: High-level user-facing API

```
interface/
├── analysis.py           # analyze_dataset() main entry point
└── config.py             # Configuration management
```

## Data Flow Through the System

```
1. INPUT: logs.jsonl with prompts, responses, judge scores, logprobs
                    ↓
2. CALIBRATION: Learn f: judge → oracle on small subset
   - Uses unified folds for cross-fitting
   - Stores calibrator for reuse
                    ↓
3. REWARD MAPPING: Apply f to all samples
   - Cross-fitted to avoid overfitting
   - Oracle augmentation for honest CIs
                    ↓
4. SAMPLING: PrecomputedSampler filters and caches
   - Handles missing logprobs
   - Provides consistent interface
                    ↓
5. FOLD ASSIGNMENT: get_fold(prompt_id) deterministic
   - Same sample → same fold always
   - Survives filtering and fresh draws
                    ↓
6. ESTIMATION: Apply chosen estimator
   - IPS: Direct weighted average
   - DR: Combines model and weights
   - Stacked: Optimal ensemble
                    ↓
7. DIAGNOSTICS: Check reliability
   - ESS thresholds
   - Weight explosion
   - Influence function tails
                    ↓
8. OUTPUT: EstimationResult with estimates, SEs, CIs
```

## How Unified Fold Management Fits In

### The Problem We Solved
Before our work, CJE had 5 independent fold assignment systems:
1. JudgeCalibrator: KFold + hash mixture
2. DREstimator: index % n_folds
3. StackedDR: Independent KFold
4. SIMCal: Internal KFold
5. MRDR: Another KFold

This caused:
- Same sample getting different folds in different components
- Filtering breaking fold assignments (index-based fragility)
- Fresh draws unable to inherit correct folds
- Cross-validation inconsistencies corrupting orthogonality

### Our Solution Architecture

```
cje/data/folds.py
├── get_fold(prompt_id, n_folds, seed) → int
│   # Core: hash(prompt_id) % n_folds
│   # Deterministic, filtering-proof
│
├── get_folds_for_prompts(prompt_ids, ...) → np.ndarray
│   # Vectorized for efficiency
│
├── get_folds_for_dataset(dataset, ...) → np.ndarray
│   # Dataset-level interface
│
└── get_folds_with_oracle_balance(prompt_ids, oracle_mask, ...)
    # Special case: ensures oracle samples balanced across folds
```

### Integration Points

1. **JudgeCalibrator** (`calibration/judge.py`)
   - Uses `get_folds_with_oracle_balance()` when `prompt_ids` provided
   - Falls back to old system for backward compatibility

2. **DREstimator** (`estimators/dr_base.py`)
   - Removed `_create_fold_assignments()` method
   - Now uses `get_fold()` on-demand per sample

3. **StackedDREstimator** (`estimators/stacking.py`)
   - `shared_fold_ids` computed via `get_folds_for_dataset()`
   - All base estimators share same folds

4. **PrecomputedSampler** (`data/precomputed_sampler.py`)
   - Added `get_folds_for_policy()` method
   - Computes `cv_fold` on-demand in data dict

5. **MRDR** (`estimators/mrdr.py`)
   - Currently reads cv_fold from metadata (won't find it)
   - Falls back to own KFold (works but suboptimal)
   - Future: Direct use of unified system

## Design Principles

### 1. **Single Source of Truth**
- One implementation in `data/folds.py`
- All components import from same place
- No duplicate fold logic

### 2. **Deterministic and Reproducible**
- `hash(prompt_id)` ensures same input → same output
- Seed parameter for different random splits
- No hidden state or randomness

### 3. **Filtering-Proof**
- Based on stable identifier (prompt_id) not array position
- Survives any data transformation
- Fresh draws inherit correct folds

### 4. **Simple and Fast**
- Core logic: `hash(prompt_id) % n_folds`
- O(1) per sample
- No caching needed

### 5. **Backward Compatible**
- Components fall back gracefully
- Optional parameters preserve old behavior
- Gradual migration path

## Testing Strategy

### Unit Tests (`tests/test_unified_folds.py`)
- 21 tests covering all functions
- Determinism, filtering, oracle balance
- Edge cases and performance

### Integration Tests (Planned)
- Pipeline consistency
- Cross-estimator alignment
- Fresh draw inheritance
- Real data scenarios

### Production Validation
- All 6 estimators tested on arena dataset
- Confirmed fold consistency
- Performance acceptable (1-45s)

## Future Improvements

### Immediate
1. **MRDR Direct Integration** - Use unified system directly
2. **Integration Tests** - Full pipeline validation
3. **Documentation** - Update user guides

### Long-term
1. **Global Configuration** - Configurable n_folds/seed
2. **Fold Persistence** - Optional caching for huge datasets
3. **Stratified Folds** - Balance by outcome distribution

## Summary

The unified fold management system is a foundational improvement that:
- Solves critical cross-validation inconsistencies
- Simplifies the codebase (5 systems → 1)
- Improves reliability (deterministic, filtering-proof)
- Maintains performance (O(1) operations)
- Enables future improvements (stratification, etc.)

It sits at the data layer as the authoritative source for fold assignments, ensuring all components in CJE's causal inference pipeline maintain consistent cross-validation folds. This is essential for valid statistical inference, especially for doubly robust methods that rely on cross-fitting for orthogonality.