# Phase 1: Dataset Preparation

This phase creates a complete dataset for evaluating CJE on ChatBot Arena prompts.

## Overview

The goal is to generate a dataset containing:
- 10,000 prompts from ChatBot Arena
- Responses from multiple policies (π₀, π_cot, π_bigger_model, π_bad)
- Oracle labels for ground truth
- Judge scores with two uncertainty methods (deterministic and confidence intervals)

## Scripts (Run in Order)

### 1. `01_prepare_data.py`
Extracts 10,000 prompts from ChatBot Arena dataset.
```bash
python 01_prepare_data.py
```
Output: `../data/arena_prompts_10k.jsonl`

### 2. `02_generate_logs.py`
Generates π₀ (logging policy) responses for all 10,000 prompts.
```bash
python 02_generate_logs.py
```
Output: `../data/p0_replies.jsonl`

### 3. `02b_generate_target_ground_truth.py`
Generates responses from target policies (π_cot, π_bigger_model, π_bad) for 500 prompts.
```bash
python 02b_generate_target_ground_truth.py
```
Output: `../data/target_ground_truth.jsonl`

### 3. `03_generate_oracle_labels.py`
Generates oracle labels using an AI judge (e.g., GPT-4).
```bash
export OPENAI_API_KEY="your-api-key"
python 04_generate_oracle_labels.py --model gpt-4o
```
Outputs:
- `../data/labeling/oracle_labels_calibration_detailed.jsonl` (2,500 labels)
- `../data/labeling/oracle_labels_validation_detailed.jsonl` (1,500 labels)

### 4. Judge Scoring (Two Methods)

#### 4a. `04a_deterministic_judge_scores.py`
Scores responses with deterministic judge (variance=0).
```bash
python 04a_deterministic_judge_scores.py
```
Output: `../data/p0_scored_deterministic.jsonl`

#### 4b. `04b_uncertainty_judge_scores.py`
Scores responses with uncertainty-aware judge (95% CI).
```bash
python 04b_uncertainty_judge_scores.py
```
Output: `../data/p0_scored_uncertainty.jsonl`

### 5. `05_finalize_dataset.py`
Validates and summarizes the complete dataset.
```bash
python 05_finalize_dataset.py
```
Output: `../data/dataset_info.json`

## Dataset Structure

```
data/
├── arena_prompts_10k.jsonl          # 10,000 prompts
├── p0_replies.jsonl                 # π₀ responses
├── target_ground_truth.jsonl        # Target policy responses
├── p0_scored_deterministic.jsonl    # Judge scores (variance=0)
├── p0_scored_uncertainty.jsonl      # Judge scores (with CI)
├── dataset_info.json                # Dataset summary
└── labeling/
    ├── oracle_labels_calibration_detailed.jsonl  # 2,500 labels
    ├── oracle_labels_validation_detailed.jsonl   # 1,500 labels
    └── oracle_labels.csv            # Combined CSV format
```

## Key Statistics

- **Total prompts**: 10,000
- **Calibration set**: 2,500 oracle-labeled π₀ responses
- **Validation set**: 1,500 oracle-labeled target policy responses (500 × 3 policies)
- **Judge scoring methods**: 2 (deterministic, uncertainty)
- **Total oracle labels**: 4,000

## Next Steps

Once the dataset is finalized, proceed to Phase 2 for CJE ablations.