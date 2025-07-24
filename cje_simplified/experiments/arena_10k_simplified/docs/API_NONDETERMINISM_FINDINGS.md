# Fireworks API Non-Determinism Findings

## Issue
During debugging of the test_e2e_pipeline, we discovered significant non-determinism in the Fireworks API logprob computations.

## Key Findings

1. **Variance in Logprob Values**
   - For the same input (prompt, response, system prompt, model, temperature=0.7), the API returns different logprob values
   - Example for "How do I make a paper airplane?" response:
     - Range: -40.8 to -43.5 (variance of ~3 points)
     - Tested with 5 consecutive API calls

2. **Outlier Values**
   - In the test pipeline, the base policy received logprob of -156.5
   - Fresh computations consistently return values around -42 to -43
   - This suggests occasional API anomalies that return extremely negative values

3. **Policy Comparison Issues**
   - Base policy logprob: -156.5 (likely an outlier)
   - Clone policy logprob: -40.3 (normal range)
   - Unhelpful policy logprob: -107.8
   - These differences make it appear that the base policy response is much less likely than it actually is

## Impact on CJE Analysis

1. **Importance Weight Calculation**
   - CJE uses importance weights: w = exp(target_logprob - base_logprob)
   - With base=-156.5 and clone=-40.3, the weight becomes exp(116.2) which is astronomically large
   - This can severely distort the CJE estimates

2. **Non-Reproducibility**
   - Results may vary between runs due to API non-determinism
   - Makes debugging and validation more difficult

## Recommendations

1. **Add Outlier Detection**
   - Flag logprobs that are unusually negative (e.g., < -100 for short responses)
   - Consider re-computing outliers or using robust statistics

2. **Multiple Samples**
   - Consider computing logprobs multiple times and using median/mean
   - Would increase API costs but improve stability

3. **Monitoring**
   - Track logprob distributions to identify anomalies
   - Log API response metadata for debugging

4. **Consider Temperature=0**
   - May reduce variance (though initial tests still showed some variance)
   - Trade-off between determinism and matching generation temperature