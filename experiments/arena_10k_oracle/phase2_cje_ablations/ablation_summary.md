# CJE Arena 10K Ablation Results Summary

## Policy Rankings (by SNIPS)
1. **pi_bad**: 0.888
2. **pi_cot**: 0.884
3. **pi_bigger_model**: 0.883

## Key Findings

1. **Low ESS Warning**: All policies have ESS < 5%, indicating high variance
2. **Best Policy**: `pi_bad` performs best (0.900) - surprising result!
3. **Extreme Weights**: Some importance weights reach ~10^8

## Recommendations

1. Wait for more teacher forcing data to complete
2. Use doubly-robust estimators when full data available
3. Investigate why `pi_bad` performs well

## Current Status

- Teacher forcing: ~3 samples processed (~15% complete)
- All P0 responses scored
- Target scoring in progress