# Phase 1: Dataset Preparation

This phase prepares the Arena 10K dataset with responses from multiple policies and judge scores using deterministic llama.cpp teacher forcing.

## 🚀 Quick Start

```bash
# 1. Install llama-cpp-python
pip install llama-cpp-python

# 2. Download model (~2.5GB)
mkdir -p ../models
curl -L -o ../models/Llama-3.2-3B-Instruct-Q6_K.gguf \
  https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q6_K.gguf

# 3. Source API keys (still needed for judge/oracle)
source /Users/eddielandesberg/PycharmProjects/causal-judge-evaluation/set_secrets.sh

# Full run with 10,000 samples (default)
python run_phase1_pipeline.py

# Test run with 5 samples
python run_phase1_pipeline.py 5

# To start completely fresh, manually clean up:
rm -rf data/
rm .pipeline_checkpoint.pkl
```

## 📋 Pipeline Steps

The `run_phase1_pipeline.py` script runs all steps automatically:

1. **Download prompts** - Sample English prompts from ChatBot Arena conversations
2. **Generate responses** - From P0 (baseline) and 2 target policies  
3. **Compute log probabilities** - P0 responses under all policies using llama.cpp
4. **Judge scoring** - Deterministic and uncertainty-based scores (writes Phase 2 format)
5. **Oracle labels** - Ground truth from GPT-4o for calibration/validation (writes to Phase 2)
6. **Validate and summarize** - Verify all files created and generate statistics

## 🎯 Output Files

### Phase 1 Data (in `data/`)
```
data/
├── arena_prompts_10k.jsonl               # Sampled prompts
├── all_responses.jsonl                   # All policy responses (consolidated)
├── logprobs.jsonl                        # P0 log probs under each policy
├── oracle_labels_calibration.jsonl       # Oracle labels for P0 (25% sample)
├── oracle_labels_validation.jsonl        # Oracle labels for targets (25% sample)
└── dataset_info.json                     # Dataset summary
```

### Phase 2 Data (in `../data/`)
Automatically created by pipeline scripts:
```
../data/
├── p0_scored_deterministic.jsonl         # P0 with scores + log probs
├── p0_scored_uncertainty.jsonl           # P0 with uncertainty + log probs
├── targets_scored_deterministic.jsonl    # Target policies with scores
├── targets_scored_uncertainty.jsonl      # Target policies with uncertainty
└── labeling/
    ├── oracle_labels_calibration_detailed.jsonl
    └── oracle_labels_validation_detailed.jsonl
```

## 📊 Policies

Using llama.cpp with Llama 3.2 3B model:

- **P0 (baseline)**: Temperature 0.5, standard helpful assistant
- **pi_clone**: Identical to P0 (for validation - weights should be exactly 1.0)
- **pi_bad**: Temperature 0.5 with unhelpful system prompt

## 🔧 Pipeline Features

- **Always Resumes**: Automatically continues from where it left off if interrupted
- **Data Integrity Checks**: Validates output files before skipping completed steps  
- **Parameter Validation**: Prevents accidental parameter mismatches
- **Consolidated Logging**: All output saved to timestamped log file
- **Manual Clean Start**: Requires explicit deletion of data/ and checkpoint file

## 📝 Individual Scripts (Advanced)

Scripts 2-5 have no arguments and use fixed settings from `arena_10k.yaml`:

```bash
# 1. Prepare prompts (requires samples/seed)
python 01_prepare_data.py --samples 5000 --seed 42

# 2-5. All other scripts (no arguments)
python 02_generate_responses.py
python 02b_compute_logprobs.py
python 03_judge_scores_deterministic.py
python 03b_judge_scores_uncertainty.py
python 04_generate_oracle_labels.py
python 05_validate_and_summarize.py
```

## ⚡ Key Design Principles

- **Minimal configuration**: Scripts 2-5 take no arguments for consistency
- **Fixed settings**: All configuration from `arena_10k.yaml`  
- **Direct Phase 2 output**: Judge scoring writes Phase 2 format directly
- **Consolidated format**: All responses in single file for easy processing
- **Automatic cleanup**: Checkpoint files removed after successful completion

## ⚠️ Important Notes

- Always source secrets before running (requires both FIREWORKS_API_KEY and OPENAI_API_KEY)
- Full 10K run takes ~50-75 hours and costs ~$70-80 total (including oracle labels)
- Phase 2 data is written directly to `../data/` by judge scoring scripts
- Monitor for 0.0 log probabilities (indicates teacher forcing bug)
- Pipeline always resumes from checkpoint if one exists
- To start fresh: `rm -rf data/ ../data/ .pipeline_checkpoint.pkl`