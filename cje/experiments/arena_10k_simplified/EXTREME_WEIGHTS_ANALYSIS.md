# Extreme Weights Analysis: Arena 10K Simplified Dataset

## Executive Summary

The arena_10k_simplified dataset exhibits **catastrophically extreme importance weights** that effectively destroy the statistical efficiency of off-policy evaluation. The Effective Sample Size (ESS) for most target policies is less than 0.5% of the actual sample size, meaning 99.5% of the data contributes negligible information.

## Key Findings

### 1. Weight Distribution Statistics

| Policy | ESS | ESS Ratio | Median Weight | % Near Zero (<0.01) | % Very Large (>100) |
|--------|-----|-----------|---------------|---------------------|---------------------|
| **clone** | 21 | 0.4% | 1.00 | 0.1% | 0.1% |
| **parallel_universe_prompt** | 1 | 0.02% | 1.64e-04 | 65.5% | 1.8% |
| **premium** | 7 | 0.1% | 2.68e-31 | 96.1% | 0.0% |
| **unhelpful** | 3 | 0.06% | 2.14e-25 | 98.4% | 0.02% |

### 2. Root Causes of Extreme Weights

#### A. Model Mismatch
- **Base policy**: Llama-3.3-70B with "You are a helpful assistant"
- **Premium policy**: Llama-3.1-405B (different model!)
- **Unhelpful policy**: Same model but adversarial prompt

The premium policy uses a completely different model (405B vs 70B), causing massive probability divergence.

#### B. Response Length Correlation
Longer responses have more extreme weights:
- **<100 chars**: 84.5% have weights <0.01
- **100-500 chars**: 97.1% have weights <0.01
- **500-1000 chars**: 98.4% have weights <0.01
- **>1000 chars**: 99.4% have weights <0.01

This is because longer sequences accumulate more log probability differences.

#### C. System Prompt Divergence
The "unhelpful" policy has a fundamentally different objective:
```
"You are an unhelpful assistant that deliberately confuses and misleads the user."
```
This creates responses that are astronomically unlikely under the base policy.

### 3. Log Probability Analysis

Base policy log probabilities show extreme variation:
- **Range**: [-446.62, 0.00]
- **Mean**: -57.54
- **Std**: 39.79

Over 50% of samples have log probabilities below -50, indicating the base policy itself assigns very low probability to many responses.

**Note on 0.0 log probabilities**: 27 samples have exactly 0.0 log probability (P=1.0), which is **credible and expected** for deterministic tasks like repetition ("Repeat: X" → "X") or single-word responses. See ZERO_LOGPROB_ANALYSIS.md for details.

### 4. Specific Examples

Example weight calculations showing the problem:

**Sample 1** (Parent wedding question):
- Base logprob: -37.15
- Unhelpful logprob: -115.82
- **Weight: 6.81e-35** (effectively zero)

**Sample 3** (Chatbot arena question):
- Base logprob: -115.97
- Premium logprob: -224.51
- **Weight: 3.72e-44** (minimum float precision)

### 5. Judge Score Relationship

**Critical finding**: There's almost no correlation between judge scores and importance weights!

| Policy | Correlation(judge, log_weight) | 
|--------|--------------------------------|
| clone | -0.004 |
| parallel_universe_prompt | -0.046 |
| premium | 0.155 |
| unhelpful | -0.093 |

This means:
- **SIMCal cannot effectively calibrate these weights** because weights don't vary systematically with judge scores
- High-quality responses (high judge scores) can still have extreme weights
- The overlap problem is orthogonal to response quality

For the "unhelpful" policy, paradoxically:
- High judge scores (>0.7) have even MORE extreme weights (median 3.98e-26)
- Low judge scores (<0.3) have less extreme weights (median 1.65e-12)
- This is backwards from what we'd expect!

## Why This Matters

### 1. Statistical Inefficiency
With ESS < 1%, we're effectively using only 1-20 samples out of 5000. This means:
- Massive variance in estimates
- Extremely wide confidence intervals
- Unstable results across runs

### 2. Numerical Instability
Many weights are at the limits of floating-point precision:
- Weights as low as 3.72e-44 (smallest representable positive float32)
- Risk of numerical underflow in computations
- Potential for catastrophic cancellation

### 3. Calibration Limitations
Even with SIMCal calibration:
- Can't fix fundamental lack of overlap
- Extreme weights remain extreme after calibration
- Variance reduction is limited when starting from such extremes

## Recommendations

### 1. Immediate Fixes
- **Remove "premium" policy** from experiments (different model = no overlap)
- **Reduce "unhelpful" adversarial strength** or remove entirely
- **Use shorter maximum response lengths** to reduce weight extremity

### 2. Data Collection Improvements
- **Use mixture base policy**: Sample from weighted mixture of all target policies
- **Ensure model consistency**: Use same base model for all policies
- **Limit response length**: Cap at 500-1000 tokens to prevent extreme accumulation

### 3. Methodological Solutions
- **Use DR estimators exclusively**: They can handle poor overlap better
- **Increase fresh draws**: More fresh samples to compensate for poor logged data
- **Set stricter ESS thresholds**: Refuse estimates when ESS < 10%

### 4. Alternative Approaches
- **Online evaluation**: Deploy policies and measure directly
- **Stratified sampling**: Oversample regions with poor overlap
- **Adaptive importance sampling**: Use mixture proposals

## Technical Details

### Weight Computation
```python
weight = exp(target_logprob - base_logprob)
```

When `target_logprob - base_logprob < -50`, weight becomes effectively zero.
When the difference exceeds -100, we hit numerical underflow.

### ESS Formula
```python
ESS = (sum(weights))^2 / sum(weights^2)
```

With extreme weights, the denominator is dominated by a few large weights, crushing ESS.

### Impact on Estimators
- **RawIPS**: Completely broken (ESS < 1%)
- **CalibratedIPS**: Slightly better but still poor
- **DR methods**: Only viable option due to fresh draws

## Practical Impact on CJE Results

Despite extreme weights, CalibratedIPS produces estimates:
- **clone**: 0.752 ± 0.004 (seems reasonable)
- **parallel_universe_prompt**: 0.751 ± 0.004 (seems reasonable)
- **premium**: 0.671 ± 0.009 (large SE, questionable)
- **unhelpful**: 0.724 ± 0.005 (surprisingly high for an adversarial policy!)

The "unhelpful" estimate of 0.724 is highly suspicious - an adversarial policy shouldn't score this well. This suggests the extreme weights are causing the estimator to rely on a tiny fraction of samples that happen to have non-zero weights, leading to biased estimates.

## Conclusion

The current dataset has **fundamental overlap problems** that no amount of calibration can fix. The policies are too divergent, particularly:
1. Using different models (premium)
2. Having adversarial objectives (unhelpful)
3. Generating long responses that accumulate probability differences

**Primary recommendation**: Redesign the experiment with policies that have reasonable overlap, or switch to online evaluation methods.

## Appendix: Detection Criteria

A dataset has "extreme weight problems" if:
- ESS < 10% for any target policy
- >50% of weights are below 0.01
- Weight range exceeds 6 orders of magnitude
- Median weight differs from 1.0 by >2 orders of magnitude

The arena_10k_simplified dataset fails ALL these criteria for 3 out of 4 target policies.