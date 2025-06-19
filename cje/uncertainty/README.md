# Uncertainty-Aware CJE Module

This module provides first-class support for uncertainty quantification in Causal Judge Evaluation (CJE). Unlike the original implementation where uncertainty was optional, this module treats uncertainty as a fundamental aspect of evaluation.

## Key Features

### 1. **Mandatory Uncertainty**
Every judge score includes both a mean and variance. Deterministic judges simply use variance=0.

```python
from cje.uncertainty import JudgeScore

# All scores have uncertainty
score = JudgeScore(mean=0.75, variance=0.03)
```

### 2. **Automatic Calibration**
The module provides two types of calibration:
- **Isotonic calibration**: Maps biased judge scores to unbiased values
- **Gamma calibration**: Corrects systematic over/under-confidence in judge uncertainty

**Important**: Gamma is computed AFTER isotonic calibration to measure only irreducible uncertainty, not bias.

```python
# Calibration order matters!
iso_model, gamma = calibrate_scores_isotonic(judge_scores, true_labels)
# gamma measures dispersion after debiasing, not total MSE
```

### 3. **Variance-Based Weight Shrinkage**
Automatically shrinks importance weights for high-uncertainty samples to improve ESS:

```python
# Optimal shrinkage: w* = w / (1 + λv)
# λ* = Cov[w²v, w(r-μ)] / E[w²v²]
```

Three shrinkage methods available:
- `"optimal"`: Uses the covariance formula
- `"adaptive"`: Maintains minimum ESS constraint
- `"fixed"`: User-specified lambda

### 4. **Multi-Policy Support**
Clean support for evaluating multiple policies simultaneously:

```python
result = estimator.fit(
    judge_scores=scores,
    oracle_rewards=rewards,
    importance_weights=weights,  # Shape: (n_samples, n_policies)
    policy_names=["GPT-3.5", "GPT-4", "Claude-3"]
)

# Rich comparison features
comparison = result.pairwise_comparison("GPT-4", "GPT-3.5")
print(f"Difference: {comparison['difference']:.4f}")
print(f"P-value: {comparison['p_value']:.4f}")
```

### 5. **Variance Decomposition**
Understand where uncertainty comes from:

```python
decomp = estimate.variance_decomposition
print(f"EIF contribution: {decomp.eif_pct:.1f}%")
print(f"Judge uncertainty: {decomp.judge_pct:.1f}%")
```

## Usage Example

```python
from cje.uncertainty import (
    UncertaintyAPIJudge,
    UncertaintyAwareDRCPO,
    UncertaintyJudgeConfig,
    UncertaintyEstimatorConfig,
)

# 1. Configure uncertainty-aware judge
judge_config = UncertaintyJudgeConfig(
    provider="openai",
    model_name="gpt-4",
    template="comprehensive_judge",
)
judge = UncertaintyAPIJudge(judge_config)

# 2. Score with uncertainty
judge_scores = judge.score_batch(samples)  # Returns List[JudgeScore]

# 3. Configure estimator
estimator_config = UncertaintyEstimatorConfig(
    k_folds=5,
    use_variance_shrinkage=True,
    shrinkage_method="adaptive",
)
estimator = UncertaintyAwareDRCPO(estimator_config)

# 4. Get results with uncertainty
result = estimator.fit(
    judge_scores=judge_scores,
    oracle_rewards=oracle_rewards,
    importance_weights=weights,
    policy_names=policy_names,
)

# 5. Analyze results
print(result.summary())
for policy in result.policies:
    print(f"{policy.name}: {policy.estimate.summary()}")
```

## Mathematical Details

### Gamma Calibration
Gamma corrects for systematic mis-calibration in judge confidence:

```
γ = Σ(r_i - m̂_i)² / Σ(v_i)
```

Where:
- `r_i`: True reward
- `m̂_i`: Calibrated (debiased) judge mean
- `v_i`: Judge's reported variance

**Key insight**: We use calibrated means `m̂_i` not raw means `m_i` to avoid double-counting bias as variance.

### Uncertainty-Aware Standard Error
The final standard error includes both EIF variance and judge uncertainty:

```
SE = sqrt(Var[ψ]/n + E[w²v]/n)
```

With optional shrinkage:
```
SE = sqrt(Var[ψ*]/n + E[w*²v]/n)
```

Where `w* = w/(1 + λv)` are the shrunk weights.

## Design Philosophy

1. **Uncertainty is not optional**: Every component expects and propagates uncertainty
2. **Type safety**: Validated dataclasses ensure correct data flow
3. **Clear separation**: Judge → Calibration → Estimation → Diagnostics
4. **Rich diagnostics**: Identify issues like variance concentration or poor calibration
5. **No backward compatibility**: Clean design without legacy constraints

## Migration from Original Implementation

If migrating from the scattered uncertainty implementation:

1. Replace `JudgeScore` with `cje.uncertainty.JudgeScore` (always has variance)
2. Replace conditional uncertainty logic with direct usage
3. Use `UncertaintyAwareDRCPO` instead of modifying standard DRCPO
4. Access results via `MultiPolicyUncertaintyResult` for rich features

## Common Gotchas

1. **Gamma interpretation**: γ > 1 means judge underestimates uncertainty (too confident about their uncertainty estimates)
2. **Shrinkage can reduce power**: While improving ESS, shrinkage introduces bias
3. **Variance bounds**: Max variance for [0,1] scores is 0.25
4. **Multi-policy weights**: Shape must be (n_samples, n_policies)