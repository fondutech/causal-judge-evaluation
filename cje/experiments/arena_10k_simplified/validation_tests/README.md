# Validation Tests

Validation scripts confirming all CJE paper features work correctly.

## Test Files

### `test_full_pipeline.py`
Complete end-to-end validation of all features:
- IIC variance reduction (3-95%)
- SIMCal variance cap enforcement
- Fresh draw auto-loading
- Oracle slice augmentation

### `test_auto_fresh_draws.py`
Verifies automatic fresh draw loading in DR estimators.

### `test_var_cap.py`
Confirms SIMCal variance constraints are enforced.

## Running Tests

To validate all features:
```bash
cd validation_tests
python test_full_pipeline.py
```

To test specific features:
```bash
python test_auto_fresh_draws.py  # Test fresh draw auto-loading
python test_var_cap.py           # Test SIMCal variance cap
```

## Expected Results

All tests should show:
- ✓ IIC diagnostics found and variance reduction achieved
- ✓ SIMCal variance constraints enforced when needed
- ✓ Fresh draws auto-loaded successfully
- ✓ Oracle slice augmentation applied with partial labels

## Created During

These tests were created during the validation phase to ensure all paper claims are supported by working code (August 2024).