# CF-bits: Information Accounting for Causal Inference

CF-bits provides an information-accounting layer that decomposes uncertainty into **identification width** (structural limits) and **sampling width** (statistical noise).

## Quick Start - Two Scenarios

**Note**: CF-bits assumes KPIs are rescaled to [0,1], so W₀=1.0 by default. For other scales, pass appropriate `w0` to `compute_cfbits()`.

### Scenario A: Fresh Draws Available (DR/TMLE)

```python
from cje.cfbits import cfbits_report_fresh_draws

# After fitting your DR/TMLE estimator with fresh draws
report = cfbits_report_fresh_draws(
    estimator,
    policy="target_policy",
    cfbits_cfg={"n_boot": 800, "random_state": 42}  # Optional config
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
    cfbits_cfg={"n_boot": 800, "random_state": 42}  # Optional config
)

if report["gates"]["state"] == "REFUSE":
    print("Cannot trust this estimate - need better overlap or fresh draws")
```

## Integration with CJE Diagnostics

CF-bits now integrates seamlessly with the CJE diagnostics system through the `CFBitsDiagnostics` dataclass:

```python
from cje.diagnostics import CFBitsDiagnostics, format_cfbits_summary

# CF-bits diagnostics are automatically computed when enabled
results = analyze_dataset("data.jsonl", estimator="calibrated-ips", compute_cfbits=True)

# Access structured CF-bits diagnostics
if results.cfbits_diagnostics:
    cfbits = results.cfbits_diagnostics["target_policy"]

    # One-line summary with actionable recommendations
    print(format_cfbits_summary(cfbits))
    # Output: CF-bits: 2.1 bits | (W=0.23) | Wid=0.15, Wvar=0.08 | Gate: WARNING | → Add labels

    # Check dominant uncertainty source
    if cfbits.needs_more_labels:
        print(f"Identification limited - add {cfbits.labels_for_wid_reduction} oracle labels")
    else:
        print(f"Sampling limited - collect {cfbits.logs_factor_for_half_bit}x more data")
```

The `CFBitsDiagnostics` dataclass provides:
- Width decomposition (wid, wvar, w_tot)
- Information content (bits_tot, bits_id, bits_var)
- Reliability gates (GOOD/WARNING/CRITICAL/REFUSE)
- Actionable budget recommendations

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
- **Budget rule**: To gain Δ bits, multiply adjusted sample size (n × IFR) by 2^(2Δ)

## Module Structure

```
cfbits/
├── core.py         # CF-bits computation, gates, budget helpers
├── sampling.py     # IFR, aESS, sampling width
├── overlap.py      # A-ESSF, BC on σ(S)
├── identification.py  # Wid Phase-1 certificate algorithm
├── config.py       # Centralized defaults and thresholds
├── aggregates.py   # Paper table generation utilities
└── playbooks.py    # Ready-to-use workflows
```

### Implemented Features
- ✅ IFR and aESS computation (with IFR_main/IFR_OUA distinction)
- ✅ Sampling width with oracle uncertainty augmentation (OUA via jackknife)
- ✅ Conservative A-ESSF and BC overlap metrics (no monotonicity assumption)
- ✅ CF-bits from widths with Wmax gating
- ✅ Reliability gates with LCB support and Wmax catastrophic detection
- ✅ Ready-to-use playbooks for common scenarios
- ✅ **Wid Phase-1 certificate**: Binned isotonic bands with Hoeffding bounds
- ✅ **Budget helpers**: Convert CF-bits to resource requirements
- ✅ **Paper aggregations**: LaTeX tables and efficiency leaderboards
- ✅ **Centralized config**: Unified thresholds and defaults

### Phase-2 Enhancements (Future)
- ⏳ PAV-with-box-constraints for Wid
- ⏳ Complete EIF for CalibratedIPS

## Budget Helpers

```python
from cje.cfbits import logs_for_delta_bits, labels_for_oua_reduction

# To gain 0.5 bits (halve width), need to multiply sample size by:
log_factor = logs_for_delta_bits(0.5)  # Returns 2.0
print(f"Need {log_factor:.1f}x more logs")

# To gain 1 bit (quarter width):
log_factor = logs_for_delta_bits(1.0)  # Returns 4.0
print(f"Need {log_factor:.1f}x more logs")

# How many labels to reduce OUA from 40% to 10%?
additional_labels = labels_for_oua_reduction(0.4, 0.1, n_samples=1000, current_labels=50)
print(f"Need {additional_labels} more oracle labels")
```

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

### Identification Width

```python
compute_identification_width(estimator, policy, alpha=0.05, n_bins=20, min_labels_per_bin=3) → (float, dict)
```
Compute structural uncertainty using Phase-1 certificate with binned isotonic bands and Hoeffding bounds.
Returns diagnostics including `p_mass_unlabeled` - the target mass on bins without oracle labels.

### CF-bits

```python
compute_cfbits(w0, wid, wvar, ifr_main=None, ifr_oua=None) → CFBits
```
Compute bits of information from width components. Prefers IFR_OUA when available.

### Gates

```python
apply_gates(aessf=None, aessf_lcb=None, ifr=None, ifr_lcb=None, tail_index=None, wid=None, wvar=None) → GatesDecision
```
Apply reliability thresholds and generate recommendations. Uses lower confidence bounds (LCBs) when available for conservative gating. Includes Wmax gating for catastrophic uncertainty (refuses if Wmax > 1.0 on [0,1] KPI scale).

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