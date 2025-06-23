# Phase 1 Data Flow

## Overview
This document describes the data flow through Phase 1 of the Arena 10K Oracle experiment.

## Input/Output Summary

### Step 1: Prepare Data (01_prepare_data.py)
- **Input**: Downloads from Hugging Face
- **Output**: `../data/arena_prompts_10k.jsonl` (10,000 prompts)

### Step 2a: Generate π₀ Responses (02a_generate_p0_responses.py)
- **Input**: `../data/arena_prompts_10k.jsonl`
- **Output**: `../data/p0_replies.jsonl` (10,000 responses with log probabilities)

### Step 2b: Generate Target Responses (02b_generate_target_responses.py)
- **Input**: `../data/arena_prompts_10k.jsonl`
- **Output**: `../data/target_responses.jsonl` (30,000 responses = 10k × 3 policies)
- **Note**: Runs all 3 target policies in parallel by default

### Step 3: Generate Oracle Labels (03_generate_oracle_labels.py)
- **Input**: 
  - `../data/p0_replies.jsonl` (samples 25% for calibration)
  - `../data/target_responses.jsonl` (samples 5% for validation)
- **Output**:
  - `../data/labeling/oracle_labels_calibration_detailed.jsonl` (2,500 labels)
  - `../data/labeling/oracle_labels_validation_detailed.jsonl` (1,500 labels)
  - `../data/labeling/oracle_labels.csv` (combined CSV format)

### Step 4a: Score π₀ Deterministic (04a_deterministic_judge_scores.py)
- **Input**: `../data/p0_replies.jsonl`
- **Output**: `../data/p0_scored_deterministic.jsonl`

### Step 4b: Score π₀ Uncertainty (04b_uncertainty_judge_scores.py)
- **Input**: `../data/p0_replies.jsonl`
- **Output**: `../data/p0_scored_uncertainty.jsonl`

### Step 4c: Score Targets Deterministic (04c_score_targets_deterministic.py)
- **Input**: `../data/target_responses.jsonl`
- **Output**: `../data/targets_scored_deterministic.jsonl`

### Step 4d: Score Targets Uncertainty (04d_score_targets_uncertainty.py)
- **Input**: `../data/target_responses.jsonl`
- **Output**: `../data/targets_scored_uncertainty.jsonl`

### Step 5: Finalize Dataset (05_finalize_dataset.py)
- **Input**: All 9 files from above steps
- **Output**: `../data/dataset_info.json` (summary statistics)

## Parallelization Opportunities

1. **Steps 2a & 2b**: Can run in parallel (different data)
2. **Step 2b**: Automatically runs all 3 policies in parallel (built-in)
3. **Steps 4a, 4b, 4c, 4d**: Can all run in parallel (independent scoring tasks)

## Key Design Decisions

1. **All 10k prompts processed**: Target responses generated for all prompts, not just a sample
2. **Oracle sampling**: Only a fraction labeled for cost efficiency (25% calibration, 5% validation)
3. **Dual scoring**: Every response scored with both deterministic and uncertainty judges
4. **Consistent paths**: All scripts use `../data/` for seamless sequential execution