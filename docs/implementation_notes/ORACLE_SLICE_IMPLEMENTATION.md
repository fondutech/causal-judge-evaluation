# Oracle Slice Augmentation Implementation

## Summary

Implemented oracle slice augmentation for CJE to provide honest confidence intervals that account for the uncertainty in learning the judge→oracle calibration map from finite oracle labels.

## Key Features

1. **Augmentation Term**: Adds (L/p) * m̂(S) * (Y - f̂(S)) to estimators
   - L: Indicator for oracle label presence
   - p: Labeling probability (oracle coverage)
   - m̂(S): Estimated E[W|S] via isotonic regression
   - Y: True oracle label
   - f̂(S): Calibrated proxy from judge scores

2. **Honest Confidence Intervals**: Standard errors properly widen when oracle coverage is low, reflecting the true uncertainty in calibration.

3. **Seamless Integration**: Works with both IPS and DR estimators through optional configuration.

## Files Added/Modified

### New Files
- `cje/calibration/oracle_slice.py`: Core implementation
  - `OracleSliceAugmentation`: Main augmentation class
  - `OracleSliceConfig`: Configuration dataclass
  
- `cje/tests/test_oracle_slice.py`: Comprehensive test suite
  - Unit tests for all functionality
  - Integration test with CalibratedIPS

- `cje/examples/oracle_slice_example.py`: Usage demonstration
  - Shows how CIs widen with less oracle coverage
  - Demonstrates variance contribution tracking

### Modified Files
- `cje/calibration/__init__.py`: Added exports for new classes
- `cje/estimators/calibrated_ips.py`: Integrated augmentation
  - Added `oracle_slice_config` parameter
  - Fits m̂(S) during estimation
  - Adds augmentation to influence functions
  
- `cje/estimators/dr_base.py`: Added DR support
  - Passes config to internal IPS estimator
  - Augments IPS correction term in DR formula
  - Includes augmentation in metadata

### Documentation Updates
- `cje/calibration/README.md`: Added oracle slice section
- `cje/estimators/README.md`: Updated feature lists
- Added usage examples and configuration details

## Usage

### Basic Usage with CalibratedIPS
```python
from cje import CalibratedIPS
from cje.calibration import OracleSliceConfig

# Configure for honest CIs
config = OracleSliceConfig(
    enable_augmentation=True,
    enable_cross_fit=True
)

# Use with estimator
estimator = CalibratedIPS(
    sampler,
    oracle_slice_config=config
)

result = estimator.fit_and_estimate()
```

### With DR Estimators
```python
from cje import DRCPOEstimator

estimator = DRCPOEstimator(
    sampler,
    oracle_slice_config=config
)
```

## Key Benefits

1. **More Honest Uncertainty**: CIs reflect true uncertainty from finite oracle labels
2. **Automatic Adjustment**: No manual tuning needed - adapts to oracle coverage
3. **Backward Compatible**: Optional feature that doesn't affect existing code
4. **Diagnostic Tracking**: Reports variance contribution from oracle uncertainty

## Implementation Details

- **MCAR Support**: Currently assumes Missing Completely At Random labeling
- **Cross-Fitting**: Supports cross-fitted m̂(S) estimation for consistency
- **Normalization**: m̂(S) normalized to mean 1 for unbiasedness
- **Isotonic Regression**: Uses monotone constraint for E[W|S] estimation

## Future Work

- **MAR Support**: Add support for Missing At Random (score-dependent labeling)
- **Bootstrap CI**: Alternative CI construction via bootstrap
- **Sensitivity Analysis**: Tools for assessing impact of MCAR assumption

## Testing

All tests pass:
```bash
poetry run pytest cje/tests/test_oracle_slice.py -v
# 10 passed
```

## Mathematical Foundation

The augmentation corrects for using proxy f̂(S) instead of true Y:
- Base estimator: E[W * f̂(S)]
- True target: E[W * Y]
- Gap: E[W * (Y - f̂(S))]
- Augmentation estimates gap using oracle slice

This provides first-order unbiasedness and proper variance accounting.