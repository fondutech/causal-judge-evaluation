# Arena 10k Simplified

Ablation study of CJE estimators on Arena competition data, demonstrating 13.9× ESS improvement with SIMCal.

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
python compute_logprobs.py --model base --output ../data/logprobs/
python compute_logprobs.py --model clone --output ../data/logprobs/
# ... repeat for all policies

# 3. Generate fresh responses for DR estimators (requires API keys)
python generate_responses.py --policy clone --n-samples 1000
# ... repeat for all policies

# 4. Add judge scores
python add_scores_with_resume.py --input ../data/responses/ --output ../data/

# 5. Create final CJE dataset
python prepare_cje_data.py --output ../data/cje_dataset.jsonl
```

## Dataset Details

**Main Dataset**: `data copy/cje_dataset.jsonl` (994 samples from 50 prompts)
**Large Dataset (WIP)**: `data/` (5000 prompts, responses being generated)
- **Policies**: 
  - `base` - Original ChatBot Arena policy (logging policy)
  - `clone` - Claude-3-Opus clone
  - `parallel_universe_prompt` - Instruction-tuned variant  
  - `premium` - High-quality responses
  - `unhelpful` - Deliberately poor (for stress testing)
- **Scores**: 
  - Judge scores (0-1) from GPT-4
  - Oracle labels (0-1) from 10k+ human votes
- **Log probabilities**: In `data copy/logprobs/` for complete dataset
- **Fresh draws**: In `data copy/responses/` for DR estimators

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
python analyze_dataset.py --data "data copy/cje_dataset.jsonl" --estimator calibrated-ips
```

## Method

SIMCal calibration process:
1. Learn judge→oracle mapping via isotonic regression (2% labels)
2. Project weights onto monotone functions of judge score
3. Cap variance increase at ρ=2  
4. Result: Smooth weights, preserved unbiasedness

## Oracle Ground Truths

Computed from 10,000+ human judgments on ChatBot Arena:
- `clone`: 0.7359
- `parallel_universe_prompt`: 0.7553  
- `premium`: 0.7399
- `unhelpful`: 0.1440

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

## Notes

- DR estimators require fresh draws in `data copy/responses/`
- The `unhelpful` policy intentionally has poor overlap to test refusal mechanisms
- Warnings about extra prompts in fresh draws are normal and handled correctly
- The refactored `ablation.py` uses CJE's `load_fresh_draws_auto()` for proper fresh draw handling

## Citation

```bibtex
@article{cje2024arena,
  title={Causal Judge Evaluation on Arena Dataset},
  author={CJE Team},
  year={2024},
  note={13.9× ESS improvement with 2% oracle labels}
}
```