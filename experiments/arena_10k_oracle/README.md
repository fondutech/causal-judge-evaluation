# Arena 10K Oracle Experiment

Validates CJE on ChatBot Arena prompts using AI oracle labels.

## Quick Start

```bash
# Prerequisites
export OPENAI_API_KEY="your-key"      # For oracle labels
export FIREWORKS_API_KEY="your-key"   # For judge scoring

# Run everything interactively
python run_pipeline.py

# Or run manually:
cd phase1_dataset_preparation
python 05_finalize_dataset.py  # After running steps 1-4

cd ../phase2_cje_ablations
python run_ablations.py
```

## Overview

**Goal**: Validate CJE (Causal Judge Evaluation) on real ChatBot Arena data with AI oracle ground truth.

**Key Innovation**: Compare deterministic judge scoring (variance=0) vs uncertainty-aware scoring (95% CI) to see if modeling judge uncertainty improves policy evaluation.

## Two-Phase Structure

### Phase 1: Dataset Preparation
One-time generation of complete dataset with responses, oracle labels, and judge scores.

### Phase 2: CJE Pipeline Ablations  
Repeatable experiments with different estimators and configurations.

## Experiment Design

- **Data**: 10,000 prompts from ChatBot Arena
- **Policies**: 
  - π₀ (logging/baseline) - GPT-3.5
  - π_cot (chain-of-thought) 
  - π_bigger_model (Llama-4-Maverick)
  - π_bad (intentionally unhelpful)
- **Oracle labels**: 4,000 total
  - 2,500 calibration (π₀)
  - 1,500 validation (target policies)
- **Judge scoring methods**:
  - Deterministic (variance=0)
  - Uncertainty (95% CI → variance)
- **Estimators**: IPW, Self-normalized IPW, ~~Doubly Robust~~ (requires target samples)

## Complete Pipeline

### Phase 1: Dataset Preparation

```bash
cd phase1_dataset_preparation

# 1. Extract prompts
python 01_prepare_data.py

# 2. Generate responses  
python 02_generate_logs.py                   # π₀ responses
python 02b_generate_target_ground_truth.py   # Target policies

# 3. Generate oracle labels
python 03_generate_oracle_labels.py --model gpt-4o

# 4. Score with judges
python 04a_deterministic_judge_scores.py     # Variance=0
python 04b_uncertainty_judge_scores.py       # 95% CI

# 5. Finalize dataset
python 05_finalize_dataset.py
```

### Phase 2: CJE Ablations

```bash
cd ../phase2_cje_ablations

# Run all 6 ablations: {IPW, DR, SNIPW} × {det, unc}
python run_ablations.py

# Results in:
# - outputs/arena_10k_{ablation_name}/
# - results/ablation_comparison.json
```

## Key Files

```
arena_10k_oracle/
├── phase1_dataset_preparation/
│   ├── 01_prepare_data.py
│   ├── 02_generate_logs.py
│   ├── 02b_generate_target_ground_truth.py
│   ├── 03_generate_oracle_labels.py
│   ├── 04a_deterministic_judge_scores.py
│   ├── 04b_uncertainty_judge_scores.py
│   └── 05_finalize_dataset.py
│
├── phase2_cje_ablations/
│   ├── run_ablations.py
│   ├── prepare_for_cje.py
│   └── analyze_oracle_labels.py
│
└── data/
    ├── arena_prompts_10k.jsonl
    ├── p0_replies.jsonl
    ├── target_ground_truth.jsonl
    ├── p0_scored_*.jsonl
    └── labeling/
        └── oracle_labels_*.jsonl
```

## Expected Results

1. **Policy ranking**: π_bigger_model > π_cot > π₀ > π_bad
2. **Uncertainty impact**: Tighter confidence intervals with CI-based scoring
3. **Oracle validation**: ~90% agreement on 1,500 validation examples

## Technical Details

- [Data Flow](DATA_FLOW.md) - Detailed pipeline documentation
- [Experiment Structure](EXPERIMENT_STRUCTURE.md) - Architecture diagrams
- [Uncertainty Guide](UNCERTAINTY_GUIDE.md) - Judge uncertainty methods
- [Estimator Notes](ESTIMATOR_NOTES.md) - IPW vs DR implementation details

## Troubleshooting

- **Missing API keys**: Set environment variables
- **Memory issues**: Reduce batch size or use sampling
- **Checkpoint/resume**: Scripts auto-resume from checkpoints