# Phase 2: CJE Pipeline Ablations

This phase runs different CJE configurations on the prepared dataset to explore how various design choices affect policy evaluation.

## Overview

With the complete dataset from Phase 1, we can now run ablations to test:
- **Judge uncertainty methods**: Deterministic (variance=0) vs Confidence Intervals
- **Estimators**: IPW, Self-normalized IPW, Doubly Robust
- **Other variations**: Calibration methods, cross-fitting folds, etc.

## Quick Start

```bash
# Ensure dataset is ready
python ../phase1_dataset_preparation/05_finalize_dataset.py

# Run all ablations
python run_ablations.py

# Or run specific ablations
python run_ablations.py --ablations baseline_deterministic,baseline_uncertainty
```

## Available Ablations

### 1. IPW Estimators
- `ipw_deterministic`: IPW with deterministic judge scores
- `ipw_uncertainty`: IPW with uncertainty-aware judge scores

### 2. Self-Normalized IPW
- `snipw_deterministic`: Self-normalized IPW with deterministic scores
- `snipw_uncertainty`: Self-normalized IPW with uncertainty scores

### 3. Doubly Robust (Currently Disabled)
DR estimators are commented out in `run_ablations.py` because they require target policy samples not present in the current dataset. To enable DR:
1. Generate target samples for each prompt (1+ per policy)
2. Score them with the judge
3. Uncomment the DR configurations in `run_ablations.py`

See [Estimator Notes](../ESTIMATOR_NOTES.md) for why DR requires target samples.

## Running Individual Ablations

Each ablation can be run directly with CJE:

```bash
# Prepare data for CJE format
python prepare_for_cje.py

# Run specific ablation
cje run --cfg-path configs --cfg-name baseline_deterministic
```

## Understanding Results

Results are saved in `../../outputs/arena_10k_{ablation_name}/`:
- `results.json`: Policy value estimates and confidence intervals
- `calibration_plot.png`: Judge calibration visualization
- `logs/`: Detailed execution logs

### Key Metrics to Compare

1. **Policy Value Estimates**: How do estimated values differ?
2. **Confidence Intervals**: Does uncertainty reduce CI width?
3. **Calibration Quality**: Which method yields better calibration?
4. **Computational Cost**: Time and resources required

## Example Results Interpretation

```json
{
  "policy_values": {
    "pi_cot": {
      "estimate": 0.723,
      "ci_lower": 0.681,
      "ci_upper": 0.765
    },
    "pi_bad": {
      "estimate": 0.312,
      "ci_lower": 0.278,
      "ci_upper": 0.346
    }
  }
}
```

- Higher estimates indicate better policies
- Narrower CIs indicate more confident estimates
- Oracle correlation shows how well CJE matches ground truth

## Adding New Ablations

Edit `run_ablations.py` to add new configurations:

```python
ABLATION_CONFIGS["new_ablation"] = {
    "description": "Description of your ablation",
    "judge_scores": "p0_scored_deterministic.jsonl",
    "estimator": "ipw",
    "calibrator": "isotonic",
    # Add other parameters...
}
```

## Troubleshooting

- **"Dataset not finalized"**: Run Phase 1's `05_finalize_dataset.py`
- **Missing scores**: Ensure both judge scoring methods completed in Phase 1
- **Config errors**: Check YAML syntax in generated configs

## Next Steps

After running ablations:
1. Review `results/ablation_comparison.json` for summary
2. Analyze which configurations work best for your use case
3. Consider running additional ablations with different parameters