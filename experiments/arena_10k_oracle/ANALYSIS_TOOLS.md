# Arena 10K Analysis Tools

This directory contains tools for analyzing the Arena 10K Oracle experiment results.

## Core Analysis Scripts

### `create_test_dataset.py`
Creates a minimal test dataset (20 samples) for testing the analysis pipeline without needing the full 10K dataset. Generates:
- Test prompts and responses
- Deterministic and uncertainty judge scores
- Mock oracle labels

Usage:
```bash
python create_test_dataset.py
```

### `run_ablation_analysis.py`
Main analysis script that runs all CJE estimators on the scored data. Uses the `PrecomputedMultiTargetSampler` to work with pre-scored data efficiently.

Features:
- Compares deterministic vs uncertainty judging
- Tests all estimators: IPS, SNIPS, CalibratedIPS, DRCPO, MRDR
- Generates comprehensive results table
- Saves detailed results to JSON

Usage:
```bash
python run_ablation_analysis.py
```

### `visualize_ablation_results.py`
Creates visualizations from the ablation analysis results:
- Bar charts comparing estimator performance across policies
- Variance comparison plots
- Summary statistics and key insights

Usage:
```bash
python visualize_ablation_results.py
```

## Verification Tests

The `verification_tests/` directory contains scripts that verify fundamental properties:
- `test_clone_policy_weights.py`: Verifies that clone policies have all weights = 1
- `test_clone_policy_estimation.py`: Verifies that clone policy IPS equals empirical mean

## Results Structure

Results are saved in `phase2_cje_ablations/results/`:
- `ablation_analysis.json`: Detailed estimation results
- `visualizations/`: Generated plots and summary
  - `estimator_comparison.png`: Bar charts by judge type
  - `variance_comparison.png`: Standard error comparison
  - `summary.json`: Key findings and insights