# Phase 2: CJE Ablations on Arena 10K Data

This directory contains scripts and configurations for running Causal Judge Estimation (CJE) ablations on the Arena 10K dataset prepared in Phase 1.

## Overview

Phase 2 evaluates different CJE estimation methods using the precomputed data from Phase 1:
- Importance Propensity Scoring (IPS) variants
- Self-Normalized IPS (SNIPS)
- Doubly Robust methods (DR-CPO, MRDR)
- Calibrated versions of these estimators

## Prerequisites

1. **Complete Phase 1**: All data files must be generated
   ```bash
   # Check that these files exist:
   ../data/p0_scored_deterministic.jsonl
   ../data/p0_scored_uncertainty.jsonl
   ../data/targets_scored_deterministic.jsonl
   ../data/targets_scored_uncertainty.jsonl
   ```

2. **API Keys**: Not required for Phase 2 (uses precomputed data)

## Quick Start

### Run Simple CJE Analysis
```bash
# Basic IPS/SNIPS analysis with weight diagnostics
python run_cje_simple.py
```

### Analyze Importance Weights
```bash
# Detailed weight diagnostics and visualizations
python analyze_weights.py
```

### Run All Ablations (if working)
```bash
# Run all configured estimators
python run_ablations.py
```

## Scripts

### `run_cje_simple.py`
- Simple, working implementation of IPS and SNIPS
- Displays results compared to oracle ground truth
- Shows basic weight statistics

### `analyze_weights.py`
- Comprehensive importance weight analysis
- Identifies problematic samples and policies
- Creates weight distribution visualizations
- Uses built-in `weight_diagnostics` module

### `run_ablations.py`
- Attempts to run all configured estimators
- Currently has integration issues with CJE library
- May need updates to work with current API

## Configuration Files

Located in `configs/ablations/`:
- `deterministic_ips.yaml` - Basic IPS with deterministic judge scores
- `deterministic_snips.yaml` - Self-normalized IPS
- `deterministic_calibrated_ips.yaml` - IPS with propensity calibration
- `deterministic_drcpo.yaml` - Doubly Robust Cross-Policy Optimization
- `deterministic_mrdr.yaml` - Multi-Robust Doubly Robust
- `uncertainty_*.yaml` - Same estimators using uncertainty-aware judge scores

## Key Findings (from 10-100 sample runs)

### Importance Weight Issues
1. **Extreme weights**: Some policies have weights ranging from 1e-22 to 65+
2. **Low ESS**: Effective Sample Size often <30% due to weight variance
3. **Model mismatch**: Different architectures (scout vs maverick) cause severe overlap issues

### Policy-Specific Issues
- **pi_clone**: Should have weights ≈1.0, but shows 8x variation
- **pi_cot**: Extreme weights up to 65x due to system prompt changes
- **pi_bigger_model**: Catastrophic weights (1e-22 to 1.0) due to model architecture difference
- **pi_bad**: High variance due to temperature=1.0 vs 0.5 for others

### Recommendations
1. Use SNIPS over IPS for more stable estimates
2. Need 10K+ samples for reasonable ESS
3. Consider better P0 design with more overlap
4. Implement weight clipping/truncation

## Typical Output

```
Weight Statistics:
  pi_clone:     Mean: 1.502, ESS: 30.4%
  pi_cot:       Mean: 9.988, ESS: 18.6%
  pi_bigger_model: Mean: 0.100, ESS: 10.0%
  pi_bad:       Mean: 3.739, ESS: 10.8%

Policy Value Estimates:
  IPS:   pi_clone: 0.987, pi_cot: 6.695, pi_bigger_model: 0.050, pi_bad: 1.869
  SNIPS: pi_clone: 0.657, pi_cot: 0.670, pi_bigger_model: 0.500, pi_bad: 0.500
  Oracle: pi_clone: 0.950, pi_cot: 0.900, pi_bigger_model: 0.283, pi_bad: 1.000
```

## Next Steps

1. **Run with more data**: 10K samples should provide much better estimates
2. **Implement weight stabilization**: Clipping, truncation, or smoothing
3. **Try adaptive importance sampling**: Better P0 design
4. **Validate doubly robust methods**: Once integration issues are resolved

## Troubleshooting

### "PrecomputedSampler has no attribute 'sample'"
- Use `run_cje_simple.py` which has the correct implementation

### Extreme weight warnings
- Expected with small samples and policy mismatches
- Will improve with larger sample sizes

### Missing data files
- Ensure Phase 1 completed successfully
- Check file paths in error messages