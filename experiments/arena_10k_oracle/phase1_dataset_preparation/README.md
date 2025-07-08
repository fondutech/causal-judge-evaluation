# Phase 1: Dataset Preparation (Simplified)

This phase prepares the Arena 10K dataset with responses from multiple policies and judge scores.

## 🚀 Quick Start

```bash
# Source API keys (REQUIRED!)
source /Users/eddielandesberg/PycharmProjects/causal-judge-evaluation/set_secrets.sh

# Test run with 5 samples
python run_phase1_pipeline.py 5

# Test run with oracle labels (requires OPENAI_API_KEY)
python run_phase1_pipeline.py 5 --with-oracle

# Full run with 10,000 samples
python run_phase1_pipeline.py 10000
```

## 📋 Pipeline Steps

The `run_phase1_pipeline.py` script runs all steps automatically:

1. **Download prompts** - Sample from ChatBot Arena conversations
2. **Generate responses** - From P0 (baseline) and 4 target policies  
3. **Compute log probabilities** - P0 responses under all policies (teacher forcing)
4. **Judge scoring** - Deterministic and uncertainty-based scores
5. **Oracle labels** (optional) - Ground truth from GPT-4 for calibration/validation
6. **Finalize dataset** - Create summary and validate completeness

## 🎯 Output Files

```
data/
├── arena_prompts_10k.jsonl               # Sampled prompts
├── all_responses.jsonl                   # All policy responses (consolidated)
├── logprobs.jsonl                        # P0 log probs under each policy
├── responses_scored_deterministic.jsonl  # Deterministic judge scores
├── responses_scored_uncertainty.jsonl     # Uncertainty judge scores
├── oracle_labels_calibration.jsonl       # Oracle labels for P0 (if --with-oracle)
├── oracle_labels_validation.jsonl        # Oracle labels for targets (if --with-oracle)
└── dataset_info.json                     # Dataset summary
```

## 📊 Policies

- **P0 (baseline)**: Simple Llama model, temperature 0.5
- **pi_clone**: Same as P0 but with explicit prompt formatting
- **pi_cot**: Chain-of-thought reasoning
- **pi_bigger_model**: Larger Llama model
- **pi_bad**: Intentionally unhelpful (for testing)

## 🔧 Individual Scripts (Advanced)

All scripts use fixed paths and settings from `arena_10k.yaml`:

```bash
# 1. Prepare prompts (only script with arguments)
python 01_prepare_data.py --samples 5000 --seed 42

# 2. Generate responses (no arguments)
python 02_generate_responses.py

# 3. Compute log probabilities (no arguments)
python 02b_compute_logprobs.py

# 4. Judge scoring (no arguments)
python 03_judge_scores_deterministic.py
python 03b_judge_scores_uncertainty.py

# 5. Oracle labels (optional, only seed argument)
python 04_generate_oracle_labels.py --seed 42

# 6. Finalize dataset (no arguments)
python 05_finalize_dataset.py
```

## ⚡ Key Features

- **Minimal configuration**: Scripts take no arguments (except prompts and oracle seed)
- **Automatic checkpointing**: Long-running scripts can resume if interrupted
- **Consolidated format**: All responses in single file for easy processing
- **Clean runs**: Data directory is cleaned on each pipeline run

## ⚠️ Important Notes

- Always source secrets before running
- Full 10K run takes ~50-75 hours and costs ~$60
- Oracle labels add ~$10-20 in OpenAI costs
- Monitor for 0.0 log probabilities (indicates teacher forcing bug)
- All data files are cleaned/recreated on each run
- Checkpoint files are automatically cleaned up after successful completion