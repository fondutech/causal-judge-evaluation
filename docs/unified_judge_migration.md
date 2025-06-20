# Unified Judge System Migration Guide

## Overview

This guide documents the migration from the dual judge system (float-based + uncertainty-aware) to a unified system where all judges return `JudgeScore` objects with mean and variance.

## Key Benefits

1. **Simplified Architecture**: Single judge interface instead of two parallel systems
2. **Uncertainty by Default**: All scores include variance (0 for deterministic judges)
3. **Better Type Safety**: No more float/object conversions
4. **Future-Proof**: Easy to add uncertainty-aware judges later
5. **Cleaner Code**: No conditional logic for uncertainty

## Architecture Changes

### Before (Dual System)
```
cje/judge/                    cje/uncertainty/
├── judges.py (float)         ├── judge.py (JudgeScore) 
├── schemas.py (normalize)    ├── schemas.py (mean+var)
├── api_judge.py (float)      ├── uncertainty_api_judge.py
└── factory.py (float only)   └── (separate ecosystem)
```

### After (Unified System)
```
cje/judge/
├── judges_unified.py (JudgeScore base)
├── schemas_unified.py (single JudgeScore with mean+variance)
├── api_judge_unified.py (all return JudgeScore)
├── factory_unified.py (supports uncertainty methods)
└── (legacy files kept for reference)
```

## New Components

### 1. Unified JudgeScore (`schemas_unified.py`)
```python
class JudgeScore(BaseModel):
    mean: float  # The score value [0, 1]
    variance: float = 0.0  # Uncertainty [0, 0.25]
    
    # Backward compatibility
    def __float__(self) -> float:
        return self.mean
```

### 2. Unified Judge Interface (`judges_unified.py`)
```python
class Judge(ABC):
    @abstractmethod
    def score(self, context: str, response: str) -> JudgeScore:
        """All judges now return JudgeScore."""
        pass
```

### 3. Enhanced JudgeFactory (`factory_unified.py`)
```python
JudgeFactory.create(
    provider="openai",
    model="gpt-4o-mini",
    uncertainty_method="structured",  # New parameter!
    # Options: "deterministic", "structured", "monte_carlo"
)
```

### 4. Storage Compatibility (`score_storage.py`)
- Handles both old (float) and new (dict) formats
- Transparent access via `ScoreCompatibilityLayer`
- Migration utilities for JSONL files

## Migration Steps

### Phase 1: Add New Components (✅ Complete)
- Created unified schemas, judges, factory
- Added storage compatibility layer
- Created migration script

### Phase 2: Update Core Pipeline
1. **Update imports** in `run_experiment.py`:
```python
from cje.judge.factory_unified import JudgeFactory
from cje.utils.score_storage import update_row_with_score
```

2. **Update judge creation**:
```python
judge = JudgeFactory.create(
    provider=provider,
    model=model,
    uncertainty_method="structured",  # or "deterministic"
    ...
)
```

3. **Update score storage**:
```python
for row, score in zip(rows, scores):
    row = update_row_with_score(row, score, "score_raw")
```

### Phase 3: Update Downstream Components
1. **Calibration**: Update to handle variance
2. **Featurizers**: Use `UnifiedScoreAugmentFeaturizer`
3. **Estimators**: Leverage variance for shrinkage

### Phase 4: Clean Up
1. Remove old uncertainty module
2. Remove legacy judge implementations
3. Update all tests

## Usage Examples

### Creating Judges

```python
# Deterministic judge (variance always 0)
judge = JudgeFactory.create(
    provider="openai",
    model="gpt-4o-mini",
    uncertainty_method="deterministic"
)

# Structured uncertainty (model estimates its own)
judge = JudgeFactory.create(
    provider="anthropic", 
    model="claude-3-haiku",
    uncertainty_method="structured",
    structured_output_schema="JudgeScore"
)

# Monte Carlo uncertainty (multiple samples)
judge = JudgeFactory.create(
    provider="fireworks",
    model="llama-v3-70b",
    uncertainty_method="monte_carlo",
    temperature=0.3,
    mc_samples=10
)
```

### Accessing Scores

```python
# Get score from any format
from cje.utils.score_storage import ScoreCompatibilityLayer

compat = ScoreCompatibilityLayer()
mean = compat.get_score_value(row, "score_raw")
variance = compat.get_score_variance(row, "score_raw") 
score = compat.get_score(row, "score_raw")  # Full JudgeScore
```

### Backward Compatibility

```python
# Wrap legacy judge
from cje.judge.judges_unified import LegacyJudgeAdapter

legacy_judge = OldStyleJudge()  # Returns float
unified_judge = LegacyJudgeAdapter(legacy_judge, assumed_variance=0.0)
```

## Migration Script

Run the automated migration:

```bash
# Dry run first
python scripts/migrate_to_unified_judges.py --dry-run

# Apply changes
python scripts/migrate_to_unified_judges.py

# Migrate only code
python scripts/migrate_to_unified_judges.py --code-only

# Migrate only data files
python scripts/migrate_to_unified_judges.py --data-only outputs/
```

## Testing

After migration:
1. Run all tests: `poetry run pytest`
2. Check score variance is captured: Look for `score_raw_variance` in outputs
3. Verify calibration still works
4. Check estimator performance

## Rollback

If issues arise:
1. Restore `.bak` files created by migration script
2. Revert git changes
3. Use legacy judge adapter as temporary fix

## Future Enhancements

1. **Variance-aware calibration**: Use uncertainty in isotonic regression
2. **Confidence intervals**: Propagate judge uncertainty through pipeline
3. **Active learning**: Use high-variance samples for oracle labeling
4. **Ensemble judges**: Combine multiple judges with uncertainty