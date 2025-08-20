# CJE Ablation Experiments

Systematic ablation studies demonstrating the value of calibrated importance sampling and doubly robust methods for off-policy evaluation.

## Quick Start

```bash
# Run all ablations (takes ~30-60 minutes)
python run_all_ablations.py

# Or run individual ablations
python oracle_coverage.py       # Effect of oracle label coverage
python sample_size.py           # Sample size scaling behavior
python estimator_comparison.py  # Compare all estimation methods
python interaction.py           # Oracle × sample size interaction

# Analyze and visualize results
python analyze_results.py
```

## What Each Ablation Tests

### 1. Oracle Coverage (`oracle_coverage.py`)
**Question**: How many oracle labels do we need for effective calibration?
- Tests: 5%, 10%, 20%, 50%, 100% oracle coverage
- Finding: 5-10% is sufficient; diminishing returns beyond 20%

### 2. Sample Size (`sample_size.py`)  
**Question**: How does performance scale with dataset size?
- Tests: n = 100, 250, 500, 1000, 2000 samples
- Finding: Cal-IPS achieves √n convergence; DR methods help at small n

### 3. Estimator Comparison (`estimator_comparison.py`)
**Question**: How much does each technique improve estimates?
- Compares: IPS, SNIPS, Cal-IPS, DR-CPO, Cal-DR-CPO, Stacked-DR
- Finding: Calibration provides 10-20× SE reduction over SNIPS

### 4. Interaction Effects (`interaction.py`)
**Question**: When is DR most valuable vs Cal-IPS alone?
- Tests: 3×3 grid of oracle coverage × sample size
- Finding: DR critical when n < 500 or oracle < 10%

## File Structure

```
ablations/
├── core/                       # Shared infrastructure
│   ├── base.py                # BaseAblation class with caching
│   ├── schemas.py             # ExperimentSpec, result schemas
│   ├── diagnostics.py         # ESS, tail index, CV metrics
│   └── gates.py               # Reliability gates and warnings
├── oracle_coverage.py          # Ablation 1: Oracle coverage
├── sample_size.py             # Ablation 2: Sample size  
├── estimator_comparison.py    # Ablation 3: Method comparison
├── interaction.py             # Ablation 4: Interaction effects
├── run_all_ablations.py      # Master runner script
├── analyze_results.py         # Analysis and visualization
└── .ablation_cache/           # Cached results (auto-created)
```

## Key Results Summary

| Method | Standard Error | ESS Improvement | Notes |
|--------|---------------|-----------------|--------|
| Raw IPS | ~75× baseline | N/A | Unusable due to extreme variance |
| SNIPS | ~0.40 | 1× | Self-normalized but uncalibrated |
| Cal-IPS | ~0.02 | 13.9× | SIMCal calibration |
| Cal-DR-CPO | ~0.01-0.02 | 13.9× | Best overall performance |
| Stacked-DR | ~0.02 | 13.9× | Optimal combination of DR methods |

## Implementation Details

### Caching System
- Results cached to `.ablation_cache/` with SHA-based keys
- Cache persists across runs - safe to interrupt and resume
- Clear cache with: `rm -rf .ablation_cache/`

### Data Requirements
- Uses `../data/cje_dataset.jsonl` (Arena 10k simplified)
- ~1000 samples with judge scores and oracle labels
- Fresh draws in `../data/fresh_draws/` for DR methods

### Computational Requirements
- Full suite: 30-60 minutes on standard laptop
- Individual ablations: 5-15 minutes each
- Memory: < 4GB RAM
- Storage: ~100MB for cached results

## Troubleshooting

**Import errors**: Run from the ablations directory:
```bash
cd cje/experiments/arena_10k_simplified/ablations
python oracle_coverage.py
```

**Missing data**: Ensure dataset exists:
```bash
ls ../data/cje_dataset.jsonl
ls ../data/fresh_draws/  # For DR methods
```

**Cache issues**: Clear and restart:
```bash
rm -rf .ablation_cache/
python run_all_ablations.py
```

## Paper Figures

Results generate figures saved to `results/`:
- `oracle_coverage_results.png` - Figure 3a in paper
- `sample_size_scaling.png` - Figure 3b in paper  
- `estimator_comparison.png` - Figure 4 in paper
- `interaction_heatmap.png` - Figure 5 in paper