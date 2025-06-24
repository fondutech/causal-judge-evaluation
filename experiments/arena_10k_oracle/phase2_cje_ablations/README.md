# Phase 2: CJE Ablations

Evaluates CJE across different judges and estimators.

## Prerequisites
Complete Phase 1 dataset preparation first.

## Quick Start

```bash
# Run all ablations
python run_ablations_full.py --yes

# Or test with dry run
python run_ablations_full.py --dry-run --yes
```

## Ablations
- **Judge types**: deterministic, uncertainty
- **Estimators**: IPS, SNIPS, CalibratedIPS, DRCPO, MRDR
- **Total**: 10 combinations

## Output
- Configs saved to `configs/ablations/`
- Results saved to `results/ablation_results.json`
- Rich table output showing performance across policies

## Analysis
Results show:
1. Impact of uncertainty quantification on calibration
2. Estimator performance across different policy qualities
3. Variance reduction from doubly-robust methods