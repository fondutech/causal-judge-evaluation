# CJE Ablation Experiments

Systematic ablation studies demonstrating the value of calibrated importance sampling and doubly robust methods for off-policy evaluation.

## Quick Start

```bash
# Run all ablations (takes ~30-60 minutes)
python run_all_ablations.py

# Or run individual ablations
python sample_size.py           # Sample size scaling behavior
python estimator_comparison.py  # Compare all estimation methods
python interaction.py           # Oracle × sample size interaction
python iic_effect.py            # IIC variance reduction analysis

# Analyze and visualize results
python analyze_results.py

# Regenerate plots from existing data (without re-running experiments)
python regenerate_plots.py --list       # See what results are available
python regenerate_plots.py all          # Regenerate all plots
python regenerate_plots.py iic_effect   # Regenerate specific ablation

# Quick regeneration for estimator comparison
python regenerate_estimator_plots.py
```

## What Each Ablation Tests

### 1. Sample Size (`sample_size.py`)  
**Question**: How does performance scale with dataset size?
- Tests: n = 100, 250, 500, 1000, 2500, 5000 samples
- Compares: RawIPS, CalibratedIPS, DRCPO, CalibratedDRCPO
- Shows progression: IPS → Cal-IPS (calibration benefit) and DRCPO → Cal-DRCPO (DR + calibration)
- Finding: Cal-IPS achieves √n convergence; DR methods help at small n; calibration crucial at all scales

### 2. Estimator Comparison (`estimator_comparison.py`)
**Question**: How much does each technique improve estimates?
- Compares: 20 estimator variants across IPS and DR families
  - IPS family: RawIPS, SNIPS, CalibratedIPS
  - DR family: DRCPO, MRDR, TMLE, StackedDR (each with ±IIC, ±Calibration)
- Tests: 4×4 grid (4 sample sizes × 4 oracle coverage levels = 16 scenarios)
- Finding: Calibration provides 10-20× SE reduction; IIC adds 3-95% additional variance reduction
- Generates policy heterogeneity heatmaps showing SE and absolute error by method × policy

### 3. Interaction Effects (`interaction.py`)
**Question**: When is DR most valuable vs Cal-IPS alone?
- Tests: 3×3 grid of oracle coverage × sample size
- Finding: DR critical when n < 500 or oracle < 10%

### 4. IIC Effect (`iic_effect.py`)
**Question**: How much does Isotonic Influence Control reduce variance?
- Tests: DR estimators (DRCPO, TMLE, MRDR) with and without IIC
- Compares: Multiple oracle coverage levels (10%, 20%, 50%)
- Shows: R² vs SE reduction relationship - variance reduction proportional to judge explainability
- Finding: 3-95% SE reduction; highest for policies well-explained by judge scores (e.g., clone: 95% reduction)

## File Structure

```
ablations/
├── core/                       # Shared infrastructure
│   ├── base.py                # BaseAblation class for shared functionality
│   ├── schemas.py             # ExperimentSpec, result schemas
│   └── diagnostics.py         # ESS, tail index, CV metrics
├── sample_size.py             # Ablation 1: Sample size  
├── estimator_comparison.py    # Ablation 2: Method comparison
├── interaction.py             # Ablation 3: Interaction effects
├── run_all_ablations.py       # Master runner script
├── analyze_results.py         # Analysis and visualization
├── regenerate_estimator_plots.py  # Regenerate plots from existing data
└── results/                   # Generated results (auto-created)
    ├── sample_size/          
    │   ├── results.jsonl
    │   └── figure_2_sample_scaling.png
    ├── estimator_comparison/ 
    │   ├── results.jsonl
    │   ├── estimator_comparison.png         # Main 16-panel comparison
    │   └── policy_heterogeneity_*.png       # Per-scenario heatmaps (32 total)
    └── interaction/          
        ├── results.jsonl
        └── figure_3_interaction.png
```

## Finding and Understanding Results

After running experiments, results are saved in the `results/` directory:

### Result Files
- **`results.jsonl`**: One JSON object per experiment containing:
  - `spec`: Experiment configuration (estimator, oracle_coverage, etc.)
  - `estimates`: Policy value estimates for each target policy
  - `standard_errors`: Uncertainty estimates
  - `diagnostics`: ESS, calibration RMSE, etc.
  - `rmse_vs_oracle`: Error compared to ground truth
  - `success`: Whether experiment completed successfully

### Visualization Options

The estimator comparison generates comprehensive visualizations:

1. **Main comparison figure** (`estimator_comparison.png`):
   - 16-panel grid (4×4) showing all scenarios
   - Each panel compares estimator performance for that scenario
   - Adaptive color scaling for readability

2. **Policy heterogeneity heatmaps** (32 total, 2 per scenario):
   - `policy_heterogeneity_n{size}_oracle{pct}pct_by_se.png` - Colored by standard error
   - `policy_heterogeneity_n{size}_oracle{pct}pct_by_abs_error.png` - Colored by absolute error
   - Shows which estimators work best for which policies
   - Includes oracle ground truth comparison when available

**Regenerating plots from existing data:**
```bash
# Quick regeneration without re-running experiments
python regenerate_estimator_plots.py

# This loads results.jsonl and creates all visualizations
# Takes ~30 seconds vs ~15 minutes for full experiments
```

### Viewing Results
```bash
# Check what results exist
ls -la results/*/

# Pretty-print a result file
python -m json.tool results/sample_size/results.jsonl | head -50

# Count experiments per ablation
wc -l results/*/results.jsonl

# Generate summary visualizations
python analyze_results.py
```

## Key Results Summary

| Method | Standard Error | ESS Improvement | Notes |
|--------|---------------|-----------------|--------|
| Raw IPS | ~75× baseline | N/A | Unusable due to extreme variance |
| SNIPS | ~0.40 | 1× | Self-normalized but uncalibrated |
| Cal-IPS | ~0.02 | 13.9× | SIMCal calibration |
| DR-CPO | ~0.01-0.02 | 13.9× | Doubly robust baseline |
| DR-CPO+IIC | ~0.01 | 15-20× | IIC reduces variance 3-95% |
| Cal-DR-CPO | ~0.01-0.02 | 13.9× | With reward calibration |
| Cal-DR-CPO+IIC | ~0.01 | 15-20× | Best single DR method |
| Stacked-DR | ~0.02 | 13.9× | Optimal combination of DR methods |
| Cal-Stacked-DR+IIC | ~0.01 | 15-20× | Best overall performance |

## Implementation Details

### Data Requirements
- Uses `../data/cje_dataset.jsonl` (Arena 10k simplified)
- ~1000 samples with judge scores and oracle labels
- Fresh draws in `../data/fresh_draws/` for DR methods

### Computational Requirements
- Full suite: 30-60 minutes on standard laptop
- Individual ablations: 5-15 minutes each
- Memory: < 4GB RAM
- Storage: ~50MB for results

## Troubleshooting

**Import errors**: Run from the ablations directory:

**Missing data**: Ensure dataset exists:
```bash
ls ../data/cje_dataset.jsonl
ls ../data/fresh_draws/  # For DR methods
```

**Re-running experiments**: Simply run the script again:
```bash
python run_all_ablations.py
```

**Regenerating plots only**: If you have results but need plots:
```bash
python regenerate_estimator_plots.py  # For estimator comparison
python analyze_results.py --figures   # For other ablations
```

**Plot display issues**: Recent fixes ensure:
- All 16 scenarios are shown (was limited to 4)
- Adaptive color scaling based on data percentiles
- Automatic text color adjustment for readability
- Directory creation before saving files

## Paper Figures

Results generate figures saved to `results/`:
- `sample_size_scaling.png` - Sample size scaling analysis
- `estimator_comparison.png` - Comprehensive estimator comparison
- `interaction_heatmap.png` - Oracle × sample size interaction analysis