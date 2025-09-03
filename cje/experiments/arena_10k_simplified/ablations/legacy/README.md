# Legacy Ablation Code

This directory contains the original ablation implementation that has been superseded by the unified experiment system.

## Contents

**Original Ablation Classes:**
- `sample_size.py` - Sample size scaling experiments
- `estimator_comparison.py` - Systematic estimator comparisons
- `interaction.py` - Oracle coverage × sample size interaction
- `iic_effect.py` - IIC (Isotonic Influence Control) effectiveness

**Runners and Utilities:**
- `run_all_ablations.py` - Sequential runner for all ablations
- `regenerate_plots.py` - Plot regeneration from saved results
- `regenerate_estimator_plots.py` - Estimator-specific plot generation

## Migration Status

All functionality from these files has been migrated to the unified system:
- Configuration: `../config.py`
- Execution: `../run.py` (UnifiedAblation class)
- Analysis: `../analyze.py`

## Why Keep?

These files are preserved for:
1. Reference during migration validation
2. Understanding original experiment design
3. Fallback if issues arise with unified system

## Note

Do not use these files for new experiments. Use the unified system in the parent directory instead.