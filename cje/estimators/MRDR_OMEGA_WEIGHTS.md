# MRDR Omega Weight Configuration

## Overview
The MRDR (Multiple Robust Doubly Robust) estimator uses omega weights (ω) to weight samples when fitting policy-specific outcome models. The choice of omega weighting scheme significantly impacts model stability and performance.

## Omega Modes

### 1. `"w"` (Default - Recommended)
- **Formula**: ω = |W|
- **Characteristics**:
  - Most stable and balanced weighting
  - Avoids extreme weight concentration
  - Positive R² values consistently
  - Lower RMSE in outcome predictions
- **When to use**: Default choice for most applications

### 2. `"w2"`
- **Formula**: ω = W²
- **Characteristics**:
  - Moderate weight concentration
  - Squares the importance weights
  - More emphasis on high-weight samples
- **When to use**: When you want moderate concentration without extremes

### 3. `"snips"` 
- **Formula**: ω = (W - 1)²
- **Characteristics**:
  - Can lead to extreme weight concentration
  - Top 10% of samples can receive 80%+ of weight mass
  - Often produces negative R² values
  - Higher RMSE due to overfitting on few samples
- **When to use**: Only with low-variance, well-behaved weights

## Empirical Comparison

Based on testing with Arena 10k data (50% oracle coverage):

| Mode | R² Range | RMSE | Top 10% Weight Concentration |
|------|----------|------|------------------------------|
| `"w"` | 0.376 to 0.404 | 0.169 | 18.2% |
| `"w2"` | 0.245 to 0.285 | 0.183 | 35.7% |
| `"snips"` | -0.355 to 0.034 | 0.224 | 84.1% |

## Why the Default Changed

Originally, MRDR used `"snips"` as the default based on theoretical properties with Hájek (mean-one) weights. However, empirical testing revealed:

1. **Weight Concentration Problem**: With `"snips"`, a small fraction of samples dominates the outcome model training
2. **Negative R² Values**: Outcome models often perform worse than mean prediction
3. **Poor Generalization**: Models overfit to the few high-weight samples

The `"w"` mode was chosen as the new default because it:
- Provides stable, positive R² values
- Distributes weight more evenly across samples
- Achieves lower prediction error (RMSE)
- Generalizes better to out-of-fold data

## Usage

```python
# Using default (recommended)
estimator = MRDREstimator(sampler, n_folds=5)  # Uses omega_mode="w"

# Explicitly setting omega mode
estimator = MRDREstimator(
    sampler, 
    n_folds=5,
    omega_mode="w2"  # or "snips" if needed
)
```

## Implementation Details

The omega weights are computed from the calibrated importance weights (W) which have mean 1.0. The weights are then:
1. Transformed according to the omega mode
2. Floored at `min_sample_weight` (default 1e-8) to avoid zero weights
3. Used as sample weights in `IsotonicRegression.fit()`

## Monitoring Weight Concentration

Check the weight diagnostics dashboard to monitor concentration:
- Look at the "Sample Efficiency" plot
- Check what percentage of weight goes to top 10% of samples
- If concentration exceeds 50%, consider switching omega modes

## References

- Original MRDR paper recommends `"snips"` for theoretical properties
- Empirical testing on real-world data shows `"w"` performs better
- See `analyze_dataset.py` diagnostics output for detailed metrics