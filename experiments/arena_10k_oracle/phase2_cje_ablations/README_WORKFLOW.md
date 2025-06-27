# Phase 2 Analysis Workflow

## Overview
Phase 2 runs ablation analysis on the scored data from Phase 1, comparing different estimators and judge types.

## Quick Start

### Option 1: Monitor and Auto-Run
```bash
# From experiments/arena_10k_oracle/
python monitor_and_run_phase2.py
```
This will:
- Monitor P0 scoring progress
- Automatically start Phase 2 when P0 completes
- Run with available judge types (deterministic and/or uncertainty)

### Option 2: Manual Run
```bash
# Check if P0 is complete
cd data
wc -l p0_scored_*.checkpoint.jsonl

# If complete (10,000 lines), run analysis
cd ../phase2_cje_ablations
python run_direct_ablations.py --judge-types deterministic uncertainty
```

## Scripts

### 1. `run_direct_ablations.py`
Direct ablation analysis using PrecomputedMultiTargetSampler.
- Bypasses the full pipeline for speed
- Works directly with scored JSONL files
- Runs all estimators: IPS, SNIPS, CalibratedIPS, DRCPO, MRDR

**Usage:**
```bash
python run_direct_ablations.py \
    --judge-types deterministic uncertainty \
    --n-bootstrap 200 \
    --output results/direct_ablation_results.json
```

### 2. `visualize_results.py`
Creates plots and analysis from ablation results.

**Usage:**
```bash
python visualize_results.py \
    --results results/direct_ablation_results.json \
    --output-dir results/visualizations
```

**Outputs:**
- `estimator_comparison.png`: Bar plots comparing estimators
- `uncertainty_impact.png`: Heatmap of uncertainty vs deterministic
- `variance_comparison.png`: Variance analysis across methods
- `summary_report.md`: Key findings and insights

### 3. `monitor_and_run_phase2.py`
Automated monitoring and execution script.

**Usage:**
```bash
# Monitor and run when ready
python monitor_and_run_phase2.py

# Run immediately if P0 is ready
python monitor_and_run_phase2.py --no-wait

# Custom check interval (seconds)
python monitor_and_run_phase2.py --check-interval 60
```

## Expected Timeline

1. **P0 Scoring**: 
   - Deterministic: ~90 minutes
   - Uncertainty: ~3 hours

2. **Phase 2 Analysis**:
   - Per ablation: ~5-10 seconds
   - Total (10 ablations): ~2 minutes

3. **Visualization**: ~10 seconds

## Interpreting Results

### Policy Expected Values
- `pi_bad`: ~0.3 (intentionally poor)
- `pi_bigger_model`: ~0.6 (moderate improvement)
- `pi_cot`: ~0.8 (best performance)

### Key Metrics
1. **Estimator Accuracy**: How close to expected values?
2. **Uncertainty Impact**: Does uncertainty improve calibration?
3. **Variance Reduction**: Which methods have lowest variance?
4. **Doubly Robust Benefits**: Do DR methods (DRCPO, MRDR) improve?

## Troubleshooting

### "Missing required files"
- Ensure Phase 1 is complete
- Check checkpoint files have 10,000 lines

### Low scores for all policies
- Verify judge scores in data files
- Check for data loading issues

### High variance
- Increase `--n-bootstrap` (default: 100)
- Consider using SNIPS or DR methods

## Next Steps

After analysis completes:
1. Review `summary_report.md` for key findings
2. Check visualizations for patterns
3. Compare with oracle labels (when available)
4. Consider running with full target data (30,000 entries)