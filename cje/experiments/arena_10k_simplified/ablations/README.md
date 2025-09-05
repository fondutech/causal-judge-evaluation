# CJE Ablation Experiments

Systematic ablation studies demonstrating the value of calibrated importance sampling and doubly robust methods for off-policy evaluation.

## Quick Start

```bash
# Run all ablation experiments (1800 total)
python run.py       # Runs with checkpoint/resume support

# Quick verification test
python test_quick.py  # Runs 5 experiments to verify parameters are working

# Analyze results
python analyze_simple.py   # Generate summary tables and basic plots

# For detailed analysis with paper comparison
python analyze_calibration_detail.py  # Statistical significance testing
```

## Current System Structure

After consolidation, all ablation code uses the unified experiment system:

```
ablations/
├── run.py                     # Main experiment runner with checkpoint/resume
├── config.py                  # Experiment configuration
├── analyze_simple.py          # Basic analysis and summary tables
├── analyze_calibration_detail.py  # Statistical significance testing
├── test_quick.py              # Quick test script (5 experiments)
├── core/                      # Infrastructure
│   ├── base.py               # BaseAblation class (fixed IIC/SIMCal bugs)
│   └── schemas.py            # Data schemas
├── results/                   # All outputs
│   ├── all_experiments.jsonl # Raw experiment results
│   ├── checkpoint.jsonl      # Progress tracking for resume
│   └── analysis/             # Tables and plots
│       ├── main_summary.csv  # Key metrics by estimator
│       └── summary_plots.png # Visualization
└── legacy/                    # Deprecated code (reference only)
```

## What Gets Tested

The unified system (`run.py`) tests all combinations of:

### Parameters
- **Estimators**: raw-ips, calibrated-ips, dr-cpo, tmle, mrdr, stacked-dr
- **Sample sizes**: 500, 1000, 2500, 5000
- **Oracle coverage**: 5%, 10%, 25%, 50%, 100%
- **SIMCal weight calibration**: On/off for all estimators (controlled by `use_calibration`)
- **IIC (Isotonic Influence Control)**: On/off for all methods (controlled by `use_iic`)
- **Seeds**: 5 random seeds per configuration

### Expected Results
- **SIMCal weight calibration**: Should improve DR methods (reduces weight variance)
- **IIC**: 3-95% variance reduction (proportional to R² between influence and judge scores)
- **DR methods**: Critical when n < 500 or oracle < 10%
- **Stacked-DR**: Most robust across scenarios

## Output Files

### Raw Results
- `results/all_experiments.jsonl`: One line per experiment with:
  - `spec`: Configuration (estimator, sample_size, oracle_coverage, etc.)
  - `estimates`: Policy value estimates
  - `standard_errors`: Uncertainty estimates
  - `rmse_vs_oracle`: Error vs ground truth
  - `orthogonality_scores`: DR orthogonality diagnostic (should be near 0)
  - `iic_diagnostics`: Per-policy R² and SE reduction from IIC
  - `ess_relative`: Effective sample size as percentage
  - `hellinger_affinity`: Structural overlap measure (higher is better)

### Analysis Outputs
- `results/analysis/main_comparison.csv`: Summary table
- `results/analysis/calibration_comparison.csv`: Calibration on/off comparison
- `results/analysis/iic_comparison.csv`: IIC on/off comparison
- `results/analysis/rmse_by_configuration.png`: Main visualization

## Running Specific Configurations

To test specific settings, modify `config.py`:

```python
EXPERIMENTS = {
    'estimators': ['calibrated-ips', 'dr-cpo'],  # Just two
    'sample_sizes': [1000],                       # Single size
    'oracle_coverages': [0.10],                   # Single coverage
    'use_calibration': [True],                    # Just with calibration
    'use_iic': [False],                          # No IIC
    'n_seeds': 2,                                 # Fewer seeds
}
```

Then run: `python run.py`

## Legacy Code

The `legacy/` directory contains the original ablation implementation for reference. This code has been fully replaced by the unified system but is preserved for validation and historical context.

## Notes

- All diagnostics now come from the main CJE library (no local duplicates)
- The system uses the Arena 10k dataset by default (see config.py to change)
- Fresh draws are auto-loaded for DR methods when available