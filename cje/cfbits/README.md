# CF-bits: Information Accounting for Causal Inference

CF-bits provides an information-accounting layer that decomposes uncertainty into **identification width** (structural limits) and **sampling width** (statistical noise).

## Quick Start - Two Scenarios

### Scenario A: Fresh Draws Available (DR/TMLE)

```python
from cje.cfbits import cfbits_report_fresh_draws

# After fitting your DR/TMLE estimator with fresh draws
report = cfbits_report_fresh_draws(
    estimator, 
    policy="target_policy",
    n_boot=800,
    random_state=42
)

print(report["summary"])  # One-line summary
print(f"Gate: {report['gates']['state']}")  # GOOD/WARNING/CRITICAL/REFUSE
```

### Scenario B: Logging-Only (IPS/Cal-IPS)

```python
from cje.cfbits import cfbits_report_logging_only

# After fitting your IPS/CalibratedIPS estimator
report = cfbits_report_logging_only(
    estimator,
    policy="target_policy", 
    n_boot=800,
    random_state=42
)

if report["gates"]["state"] == "REFUSE":
    print("Cannot trust this estimate - need better overlap or fresh draws")
```

## Manual API (Full Control)

```python
from cje import load_dataset_from_jsonl, calibrate_dataset
from cje.data.precomputed_sampler import PrecomputedSampler
from cje.estimators.calibrated_ips import CalibratedIPS
from cje.cfbits import compute_ifr_aess, compute_sampling_width, estimate_overlap_floors

# Load and calibrate data
dataset = load_dataset_from_jsonl("data.jsonl")
calibrated_dataset, _ = calibrate_dataset(dataset, judge_field="judge_score")
sampler = PrecomputedSampler(calibrated_dataset)

# Fit estimator
estimator = CalibratedIPS(sampler)
result = estimator.fit_and_estimate()

# Compute CF-bits metrics
policy = "target_policy"

# 1. Sampling width (compute first to get var_oracle)
wvar, var_components = compute_sampling_width(estimator, policy)
print(f"Sampling width: {wvar:.3f}")

# 2. Efficiency metrics (IFR and aESS) using actual var_oracle
if_values = estimator.get_influence_functions(policy)
efficiency = compute_ifr_aess(if_values, var_oracle=var_components.var_oracle)
print(f"IFR: {efficiency.ifr_main:.2%}, aESS: {efficiency.aess_main:.1f}")
print(f"IFR (OUA): {efficiency.ifr_oua:.2%}, aESS (OUA): {efficiency.aess_oua:.1f}")

# 3. Overlap floors
weights = estimator.get_weights(policy) or sampler.compute_importance_weights(policy)
judge_scores = sampler.get_judge_scores()
overlap = estimate_overlap_floors(judge_scores, weights)
print(f"A-ESSF: {overlap.aessf:.2%}, BC: {overlap.bc:.2%}")
```

## Core Concepts

### Information Fraction Ratio (IFR)
- **What**: Ratio of efficient IF variance to actual IF variance
- **Two versions**:
  - `IFR_main`: Var(EIF)/Var(IF) - standard efficiency
  - `IFR_OUA`: Var(EIF)/(Var(IF) + n×Var_oracle) - accounts for oracle uncertainty
- **Range**: (0, 1] where 1 = perfectly efficient
- **Interpretation**: How close your estimator is to the efficiency bound

### Adjusted ESS (aESS)
- **What**: n × IFR
- **Two versions**: `aESS_main` and `aESS_OUA` corresponding to the IFR versions
- **Interpretation**: Equivalent sample size if you had an efficient estimator
- **Example**: n=1000, IFR=0.5 → aESS=500 (half as efficient as possible)

### A-ESSF (Adjusted ESS Fraction)
- **What**: Structural overlap on judge marginal σ(S)
- **Formula**: 1/(1 + χ²_S) where χ²_S = E[ω(S)²] - 1
- **Interpretation**: Upper bound on achievable efficiency via S-calibration

### CF-bits
- **Formula**: bits = log₂(W₀/W)
- **Interpretation**: Each bit represents a halving of uncertainty width
- **Decomposition**: bits_tot from (Wid + Wvar)

## Module Structure

```
cfbits/
├── core.py         # CF-bits computation and gates
├── sampling.py     # IFR, aESS, sampling width
├── overlap.py      # A-ESSF, BC on σ(S)
└── identification.py  # Wid computation (placeholder)
```

### Implemented Features
- ✅ IFR and aESS computation (with IFR_main/IFR_OUA distinction)
- ✅ Sampling width with oracle uncertainty augmentation (OUA via jackknife)
- ✅ Conservative A-ESSF and BC overlap metrics (no monotonicity assumption)
- ✅ CF-bits from widths
- ✅ Reliability gates with LCB support
- ✅ Ready-to-use playbooks for common scenarios

### Not Yet Implemented
- ⏳ Full identification width (Wid) with isotonic bands
- ⏳ Complete EIF for CalibratedIPS

## API Reference

### Efficiency Metrics

```python
compute_ifr_aess(phi, eif=None, n=None, var_oracle=0.0) → EfficiencyStats
```
Compute Information Fraction Ratio and adjusted ESS. Returns both IFR_main and IFR_OUA.

### Sampling Width

```python
compute_sampling_width(estimator, policy, alpha=0.05, use_iic=True, compute_oua=True) → (float, SamplingVariance)
```
Compute statistical uncertainty width with optional IIC variance reduction and oracle uncertainty.

### Overlap Metrics

```python
estimate_overlap_floors(S, W, method="conservative", n_boot=500) → OverlapFloors
```
Estimate structural overlap bounds on judge marginal. Uses conservative estimation without monotonicity assumptions.

### CF-bits

```python
compute_cfbits(w0, wid, wvar, ifr_main=None, ifr_oua=None) → CFBits
```
Compute bits of information from width components. Prefers IFR_OUA when available.

### Gates

```python
apply_gates(aessf=None, aessf_lcb=None, ifr=None, ifr_lcb=None, tail_index=None) → GatesDecision
```
Apply reliability thresholds and generate recommendations. Uses lower confidence bounds (LCBs) when available for conservative gating.

## Interpretation Guide

### IFR Values
- **> 0.8**: Excellent efficiency
- **0.5 - 0.8**: Good efficiency
- **0.2 - 0.5**: Moderate efficiency (consider improvements)
- **< 0.2**: Poor efficiency (investigate causes)

### A-ESSF Values
- **> 0.5**: Excellent structural overlap
- **0.2 - 0.5**: Good overlap
- **0.05 - 0.2**: Marginal (use DR methods)
- **< 0.05**: Catastrophic (consider different policies)

### CF-bits
- **0 bits**: No uncertainty reduction
- **1 bit**: Uncertainty halved
- **2 bits**: Uncertainty quartered
- **3+ bits**: Substantial reduction (8x or more)

## Mathematical Foundation

The key insight is that total uncertainty decomposes as:
```
W_tot = W_id + W_var
```

Where:
- **W_id**: What cannot be determined from logs (structural)
- **W_var**: Statistical noise from finite samples

This decomposition is auditable, with each component traceable to specific assumptions and data limitations.

## Troubleshooting

### "No influence functions available"
- Ensure estimator has been fitted: `estimator.fit_and_estimate()`
- Check that estimator stores IFs (all CJE estimators should)

### "IFR > 1.0"
- Numerical issue - IFR is theoretically bounded by 1
- Check for very small variances causing instability

### "A-ESSF > BC²"
- Theoretical violation suggesting numerical issues
- Increase bootstrap samples or check weight distribution

## References

- CF-bits paper: [Coming soon]
- CJE paper: [Calibrated Off-Policy Evaluation]
- Efficiency theory: Semiparametric efficiency bounds