# Unified Judge System - Implementation Summary

## What We Built

We've created a unified judge system that makes uncertainty a first-class citizen throughout CJE. All judges now return `JudgeScore` objects with both mean and variance, eliminating the dual system complexity.

## Key Components Created

### 1. Core Infrastructure
- **`cje/judge/schemas_unified.py`**: Single `JudgeScore` schema with mean + variance
- **`cje/judge/judges_unified.py`**: Base `Judge` interface returning `JudgeScore`
- **`cje/judge/api_judge_unified.py`**: API judges with 3 uncertainty methods
- **`cje/judge/factory_unified.py`**: Enhanced factory supporting uncertainty methods
- **`cje/judge/cached_judge_unified.py`**: Caching that preserves variance

### 2. Storage & Compatibility
- **`cje/utils/score_storage.py`**: Handles both float and structured score formats
- **`ScoreCompatibilityLayer`**: Transparent access to scores regardless of format
- Storage format: `{"mean": 0.7, "variance": 0.02}` with backward compatibility

### 3. Feature Integration
- **`cje/estimators/score_featurizer_unified.py`**: Extracts mean, variance, and std as features
- Auto-detects format and uses appropriate featurizer

### 4. Migration Tools
- **`scripts/migrate_to_unified_judges.py`**: Automated code and data migration
- **`docs/unified_judge_migration.md`**: Comprehensive migration guide

## Benefits Achieved

### 1. **Simplified Architecture**
- Single judge interface instead of two parallel systems
- No more conditional logic for uncertainty
- Cleaner imports and fewer modules

### 2. **Better Uncertainty Support**
```python
# Three ways to estimate uncertainty
judge = JudgeFactory.create(
    provider="openai",
    model="gpt-4o-mini",
    uncertainty_method="deterministic"  # Always variance=0
)

judge = JudgeFactory.create(
    provider="anthropic",
    model="claude-3-haiku",
    uncertainty_method="structured"  # Model estimates uncertainty
)

judge = JudgeFactory.create(
    provider="fireworks", 
    model="llama-v3-70b",
    uncertainty_method="monte_carlo",  # Sample multiple times
    temperature=0.3,
    mc_samples=10
)
```

### 3. **Type Safety**
- No more float/object conversions
- All judges return the same type
- Better IDE support and fewer runtime errors

### 4. **Backward Compatibility**
- `JudgeScore.__float__()` returns mean for legacy code
- Storage layer handles both formats transparently
- `LegacyJudgeAdapter` wraps old judges

## Usage Examples

### Creating a Judge
```python
from cje.judge.factory_unified import JudgeFactory

# Default: structured uncertainty
judge = JudgeFactory.create(
    provider="openai",
    model="gpt-4o-mini"
)

# Scoring returns JudgeScore
score = judge.score("What is 2+2?", "4")
print(f"Mean: {score.mean}, Variance: {score.variance}")
```

### Accessing Scores in Data
```python
from cje.utils.score_storage import ScoreCompatibilityLayer

compat = ScoreCompatibilityLayer()

# Works with any storage format
mean = compat.get_score_value(row, "score_raw")
variance = compat.get_score_variance(row, "score_raw")
```

### Storing Scores
```python
from cje.utils.score_storage import update_row_with_score

# Automatically stores in unified format
row = update_row_with_score(row, score, "score_raw")
# Creates: score_raw, score_raw_float, score_raw_variance
```

## Next Steps

### Immediate
1. Run migration script on main codebase
2. Update `run_experiment.py` to use unified judges
3. Test with existing experiments

### Future Enhancements
1. **Variance-aware calibration**: Use uncertainty in isotonic regression
2. **Confidence intervals**: Propagate uncertainty through pipeline
3. **Active learning**: Select high-variance samples for oracle labeling
4. **Ensemble judges**: Combine multiple judges weighted by confidence

## Migration Checklist

- [x] Create unified schemas and interfaces
- [x] Implement uncertainty-aware API judges
- [x] Build storage compatibility layer
- [x] Create enhanced judge factory
- [x] Add variance-aware featurizer
- [x] Write migration script
- [x] Document migration process
- [ ] Migrate run_experiment.py
- [ ] Update all tests
- [ ] Remove legacy uncertainty module
- [ ] Clean up old implementations

## Summary

The unified judge system eliminates the complexity of maintaining two parallel judge implementations. By making uncertainty mandatory (with variance=0 for deterministic judges), we get a cleaner, more theoretically sound system that's easier to extend and maintain. The migration path is straightforward with good backward compatibility, making this a low-risk, high-reward refactoring.