# Arena 10k Simplified

Ablation study of CJE estimators on simulated competition data, demonstrating 13.9× ESS improvement with SIMCal.

## Quick Start

```bash
# Run ablation study
python ablation.py --estimators calibrated-ips raw-ips dr-cpo --oracle-coverages 0.1 0.5 1.0

# Generate plots
python plot.py --results ablation_results/ablation_results.jsonl
```

## Files

- `ablation.py` - Run ablation experiments with all CJE estimators
- `analyze_dataset.py` - Direct CJE analysis with detailed diagnostics  
- `plot.py` - Generate visualization plots
- `experiment_config.py` - Policy definitions and experiment parameters
- `generate_arena_data.py` - Main data generation orchestrator
- `analysis/` - Modular analysis components used by analyze_dataset.py
- `data_generation/` - Scripts to reproduce dataset from scratch
  - `compute_logprobs.py` - Compute log probabilities with teacher forcing (supports multi-pass)
  - `generate_additional_passes.py` - Orchestrate multiple passes for non-determinism analysis
  - `prepare_cje_data.py` - Combine responses and logprobs into final dataset
  - `generate_responses.py` - Generate fresh responses for DR estimators
  - `add_scores_with_resume.py` - Add judge/oracle scores with resume capability
- `data copy/` - Complete dataset with 994 Arena samples (50 prompts, verified and in git)
- `data/` - Work-in-progress larger dataset (5000 prompts, expensive API calls!)

## Key Results

| Method | ESS | Error vs Oracle |
|--------|-----|-----------------|
| **CalibratedIPS** | 62.7% | 0.038 |
| **RawIPS** | 4.5% | 0.175 |

**Impact**: 13.9× better ESS, 4.5× lower error, works with 2% oracle labels (20 samples)

## Data Generation Pipeline

To reproduce the dataset from scratch:

```bash
cd data_generation/

# 1. Prepare prompts and base responses
python prepare_arena_data.py

# 2. Compute log probabilities for all policies
python compute_logprobs.py --responses-dir ../data/responses --output-dir ../data/logprobs/
# This computes pass 1 (original) for all policies

# 3. Generate fresh responses for DR estimators (requires API keys)
python generate_responses.py --policy clone --n-samples 1000
# ... repeat for all policies

# 4. Add judge scores
python add_scores_with_resume.py --input ../data/responses/ --output ../data/

# 5. Create final CJE dataset
python prepare_cje_data.py --output ../data/cje_dataset.jsonl

# 6. (Optional) Generate multiple passes to study API non-determinism
source ../../../set_secrets.sh  # REQUIRED: Load API keys
python generate_additional_passes.py --data-dir ../data --n-passes 5
```

## Dataset Details

**Main Dataset**: `data/cje_dataset.jsonl` (4950 samples from 4950 prompts)
- **Policies**: 
  - `base` - Llama-70B with standard helpful assistant prompt (logging policy)
  - `clone` - Same model and prompt as base (for control/comparison)
  - `parallel_universe_prompt` - Llama-70B with parallel universe system prompt
  - `premium` - Llama-405B with standard helpful assistant prompt
  - `unhelpful` - Llama-70B with deliberately unhelpful system prompt (stress testing)
- **Scores**: 
  - Judge scores (0-1) from GPT-4.1-nano
  - Oracle labels (0-1) from GPT-5 (simulated ground truth)
- **Log probabilities**: In `data/logprobs/`
- **Fresh draws**: In `data/responses/` for DR estimators

**Note**: `unhelpful` has catastrophic overlap (ESS < 1%), returns NaN by design

## Example Commands

```bash
# Single experiment
python ablation.py --estimators calibrated-ips --oracle-coverages 0.5 --n-seeds 1

# Full ablation grid  
python ablation.py \
    --estimators raw-ips calibrated-ips dr-cpo mrdr tmle \
    --oracle-coverages 0.05 0.1 0.2 0.5 1.0 \
    --sample-fractions 0.1 0.2 0.5 1.0 \
    --n-seeds 10

# Generate all plots
python plot.py --results ablation_results/

# Analyze with detailed diagnostics
python analyze_dataset.py --data "data/cje_dataset.jsonl" --estimator calibrated-ips
```

## Method

SIMCal calibration process:
1. Learn judge→oracle mapping via isotonic regression (2% labels)
2. Project weights onto monotone functions of judge score
3. Cap variance increase at ρ=2  
4. Result: Smooth weights, preserved unbiasedness

## Oracle Ground Truths

Simulated ground truth labels from GPT-5 oracle model (see experiment_config.py for details):
- These values depend on the specific dataset and oracle calibration
- Run `python analyze_dataset.py` to see current oracle estimates
- The `unhelpful` policy typically scores very low (< 0.2) by design

## Requirements

```bash
# Install CJE library
pip install -e ../../../

# Additional dependencies
pip install pandas numpy matplotlib seaborn

# For data generation (optional)
export OPENAI_API_KEY=your_key
export ANTHROPIC_API_KEY=your_key
```

## Output

Results saved to `ablation_results/`:
- `ablation_results.jsonl` - One result per line
- `ablation_results_final.json` - Complete results

Plots saved to current directory or specified `--output`.

## Known Issues

### Log Probability Issues
- **~1% null logprobs**: API returns mathematically impossible positive values for some samples
- **~18% suspicious values**: Long responses have unrealistically high log probabilities
- **Root cause**: Fireworks API bugs with teacher forcing
- **Solution**: Use multi-pass generation to identify and document non-determinism

### Multi-Pass Generation
Generate multiple passes to study API non-determinism and improve data quality:

```bash
# IMPORTANT: Must load API keys first!
source ../../../set_secrets.sh

# Generate passes 2-5 for all policies
python data_generation/generate_additional_passes.py --data-dir data --n-passes 5

# Run specific passes in parallel
python data_generation/generate_additional_passes.py \
    --data-dir data \
    --n-passes 5 \
    --parallel \
    --max-workers 4

# Analyze non-determinism (coming soon)
python data_generation/analyze_nondeterminism.py --data-dir data
```

Pass files are named: `{policy}_logprobs_pass{N}.jsonl` where N=2,3,4,5...

## Notes

- DR estimators require fresh draws in `data/responses/`
- The `unhelpful` policy intentionally has poor overlap to test refusal mechanisms
- Warnings about extra prompts in fresh draws are normal and handled correctly
- The refactored `ablation.py` uses CJE's `load_fresh_draws_auto()` for proper fresh draw handling
- Multiple passes help identify API non-determinism (observed ~6% variance between passes)

## Citation

```bibtex
@article{cje2024,
  title={Causal Judge Evaluation: Ablation Study},
  author={CJE Team},
  year={2024},
  note={13.9× ESS improvement with 2% oracle labels}
}
```