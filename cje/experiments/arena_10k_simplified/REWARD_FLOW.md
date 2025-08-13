# Reward Flow Documentation

## Overview
This document explains how rewards are handled in the CJE analysis pipeline to prevent common mistakes.

## Reward Sources (Mutually Exclusive)

1. **Pre-computed Rewards** (`reward` field exists in dataset)
   - Use as-is, no calibration needed
   - `cal_result = None`

2. **Oracle Direct** (100% oracle coverage)
   - Oracle labels used directly as rewards
   - Preserves all unique values (e.g., 25 unique values)
   - `cal_result = None` 
   - **NEVER calibrate these**

3. **Calibrated** (< 100% oracle coverage)
   - Judge scores calibrated to partial oracle labels
   - Reduces unique values (e.g., 25 → 10)
   - `cal_result = CalibrationResult(...)`
   - DR estimators may use the calibrator

## Common Mistakes to Avoid

### ❌ DON'T: Re-calibrate oracle direct rewards
```python
# WRONG - loses information!
if not cal_result:
    calibrated_dataset, cal_result = calibrate_dataset(...)
```

### ✅ DO: Check the context first
```python
# RIGHT - only calibrate if actually needed
if args.oracle_coverage < 1.0 and not cal_result:
    calibrated_dataset, cal_result = calibrate_dataset(...)
```

## Decision Tree

```
Dataset loaded
    ↓
Has rewards? → Yes → Use pre-computed
    ↓ No
Oracle coverage = 100%? → Yes → Use oracle direct (DON'T CALIBRATE!)
    ↓ No
Calibrate with partial oracle
```

## Validation

The `validate_no_unnecessary_calibration()` function will catch mistakes:
- Throws error if trying to calibrate oracle-direct rewards
- Warns about other suspicious patterns

## Testing

Always test with these cases:
1. `--oracle-coverage 1.0` (should use oracle directly)
2. `--oracle-coverage 0.5` (should calibrate)
3. Pre-computed rewards (should skip both)