# Information-Dense Paper Tables

This module generates high information-density tables optimized for paper presentation, focusing on three key questions:

1. **Who wins overall (and why)?**
2. **Which design choices matter, and in what data regimes?**
3. **Are claims statistically and numerically well-founded?**

## Usage

Generate all tables:
```bash
python generate_paper_tables.py --results results/all_experiments.jsonl --output tables/
```

## Core Tables (Main Text)

### Table 1: Estimator Leaderboard
**Goal:** One glance shows overall trade-offs among accuracy, calibration, sharpness, and ranking quality.

**Metrics:**
- **RMSE^d**: Oracle-noise-debiased RMSE (point accuracy)
- **IntervalScore^OA**: Oracle-adjusted interval score (CI quality)
- **CalibScore**: Mean |coverage - 95%| (calibration)
- **SE GeoMean**: Geometric mean of SEs (sharpness)
- **Kendall τ**: Rank correlation (policy ordering)
- **Top-1 Acc**: % correctly identifying best policy

### Table 2: Design Choice Effects (Δ Tables)
**Goal:** Make causal reading of ablations trivial with matched-pair deltas.

**Panels:**
- **A: Weight Calibration (SIMCal)**: Δ(calibrated - uncalibrated)
- **B: IIC Effect**: Δ(IIC on - IIC off)

Shows changes with bootstrap CIs and significance markers.

### Table 3: Stacked-DR Efficiency & Stability
**Goal:** Justify stacking approach and regularization.

**Metrics:**
- **SE Ratio**: SE(stacked) / min{SE(components)} (efficiency)
- **Min Eig(Σ)**: Pre/post regularization (stability)
- **Cond(Σ)**: Condition number (numerical health)
- **% Near-Singular**: Share of problematic cases
- **Runtime**: Oracle IC vs complex CV comparison

## Appendix Tables

### Table A1: Quadrant Leaderboard
RMSE^d and CalibScore by data regime (SL/SH/LL/LH).

### Table A2: Bias Patterns
Mean bias, |bias|, and per-policy biases with t-statistics.

### Table A3: Overlap & Tail Diagnostics
ESS%, tail index, Hellinger affinity bucketed as Good/OK/Poor.

### Table A4: Oracle Adjustment Share
Proportion of uncertainty from calibration; coverage with/without OA.

### Table A5: Calibration Boundary Analysis
Distance to boundaries; outlier detection rates (especially unhelpful).

### Table A6: Runtime & Complexity
Median runtime, computational complexity, folds used.

## Key Design Principles

1. **Information Density**: Every column answers a distinct question
2. **Orthogonal Metrics**: Avoid redundant columns (e.g., both CI width and SE)
3. **Geometric Means**: For SE and interval scores (robust to outliers)
4. **Paired Deltas**: Control for confounders with matched experiments
5. **Statistical Rigor**: Bootstrap CIs, Wilcoxon tests, t-statistics
6. **Visual Hierarchy**: Bold best, underline second-best in LaTeX

## Metric Definitions

### Debiased RMSE
```
RMSE^d = sqrt(MSE - oracle_variance)
```
Removes irreducible oracle noise for fair comparison.

### Interval Score (OA)
```
IS = width + (2/α) × coverage_penalty
```
Balances sharpness and calibration in one metric.

### Calibration Score
```
CalibScore = |empirical_coverage - 0.95|
```
Distance from target coverage (lower is better).

### Oracle Adjustment (OA)
```
SE^OA = sqrt(SE^2 + oracle_uncertainty^2)
```
Accounts for calibration uncertainty in very efficient estimators.

## Implementation Notes

- Results must include `robust_confidence_intervals` or fall back to SEs
- Quadrant classification uses size/oracle thresholds (1000/0.1)
- Paired deltas match on (seed, sample_size, oracle_coverage)
- Bootstrap uses 1000 replicates for CI construction
- Wilcoxon test requires n>5 and non-constant differences

## Output Formats

- **LaTeX**: Publication-ready with `\textbf{}` and `\underline{}`
- **Markdown**: GitHub-compatible tables
- **DataFrame**: For further analysis in Python

## File Structure
```
reporting/
├── __init__.py          # Module exports
├── paper_tables.py      # Core tables (1-3)
├── appendix_tables.py   # Diagnostic tables (A1-A6)
└── README.md           # This file
```