# Arena 10K Simplified Experiment

This experiment demonstrates the CJE methodology on Arena-style data using the new simplified architecture.

## Overview

The experiment follows the complete CJE pipeline:
1. Generate responses from different policies
2. Compute log probabilities using teacher forcing
3. Add judge scores (and optionally oracle labels)
4. Run CJE analysis with different reward workflows

## Workflow

### 1. Prepare Arena Data

Extract unique first-turn prompts from ChatBot Arena conversations:

```bash
python prepare_arena_data.py --samples 1000 --output data/prompts.jsonl
```

Key insights from the old codebase:
- **Deduplication is critical**: We extract unique prompts to ensure proper policy comparison
- **Fresh responses only**: Empty responses force generation from our specified policies
- **Simple filtering**: Basic language filter to get English prompts

### 2. Generate Responses
```bash
# Generate responses from all policies (base, clone, unhelpful)
python generate_responses.py --prompts data/prompts.jsonl --output-dir data/responses

# Or limit to a small test set
python generate_responses.py --prompts data/prompts.jsonl --output-dir data/responses --max-responses 10
```

This generates responses using the Fireworks API with different system prompts:
- **base**: "You are a helpful assistant."
- **clone**: "You are a helpful assistant." (same as base for comparison)
- **unhelpful**: "You are an unhelpful assistant that deliberately confuses and misleads the user."

### 3. Compute Log Probabilities
```bash
# Compute log probs for all responses under each policy's model
# This will compute log P(response|prompt) for each response under each model
python compute_logprobs.py --responses-dir data/responses --output-dir data/logprobs
```

### 4. Prepare CJE Dataset
```bash
# Create dataset with judge scores
python prepare_cje_data.py --logprobs-dir data/logprobs --output data/cje_dataset.jsonl

# Optionally add oracle labels for calibration (10% of data)
python prepare_cje_data.py --logprobs-dir data/logprobs --output data/cje_dataset.jsonl --add-oracle
```

### 5. Run CJE Analysis

The new architecture supports three distinct workflows:

#### Option A: Oracle Labels as Rewards
If you have oracle labels for all data points:
```bash
python run_cje_analysis.py --data data/cje_dataset_oracle.jsonl --use-oracle
```

#### Option B: Judge Score Calibration (Recommended)
If you have judge scores and oracle labels for a subset:
```bash
python run_cje_analysis.py --data data/cje_dataset.jsonl
```

#### Option C: Pre-calibrated Rewards
If rewards are already calibrated and included in the dataset:
```bash
python run_cje_analysis.py --data data/cje_dataset_calibrated.jsonl
```

## Data Format

The expected JSONL format after prepare_cje_data.py:
```json
{
  "prompt": "What is the capital of France?",
  "response": "The capital of France is Paris.",
  "base_policy_logprob": -15.234,
  "target_policy_logprobs": {
    "improved_v1": -12.456,
    "improved_v2": -13.789
  },
  "judge_score": 0.85,
  "oracle_label": 0.90  // Optional, only for some samples
}
```

Note: The `reward` field is not required during data loading. It will be:
- Set directly from oracle_label (Option A)
- Computed via calibration (Option B)  
- Already present in the data (Option C)

## Key Architectural Changes

1. **Decoupled Loading**: Data is loaded without requiring rewards
2. **Optional Rewards**: The `Sample.reward` field is now Optional[float]
3. **Validation at Estimation**: PrecomputedSampler validates rewards exist
4. **Flexible Workflows**: Support for oracle, calibration, and pre-calibrated data

## Example Output

```
Running CJE Analysis
==================================================

1. Loading dataset...
   ✓ Loaded 1000 samples
   ✓ Target policies: ['improved_v1', 'improved_v2']

2. Calibrating judge scores to oracle labels...
   ✓ Calibrated using 100 oracle samples
   ✓ Calibration RMSE: 0.082
   ✓ Coverage (±0.1): 89.0%

3. Running CJE estimation...

4. Results:
   ----------------------------------------
   improved_v1:
     Estimate: 0.723
     Std Error: 0.015
     95% CI: [0.694, 0.752]
   improved_v2:
     Estimate: 0.681
     Std Error: 0.018
     95% CI: [0.646, 0.716]

   🏆 Best policy: improved_v1

5. Weight diagnostics:
   improved_v1:
     Mean weight: 1.000
     Max weight: 3.421
     ESS fraction: 68.5%
   improved_v2:
     Mean weight: 1.000
     Max weight: 2.876
     ESS fraction: 74.2%

✓ Analysis complete!
```

## Troubleshooting

1. **"No valid samples could be created from data"**
   - Ensure your data has all required fields
   - Check that log probabilities are not None

2. **"PrecomputedSampler requires all samples to have rewards"**
   - You need to either calibrate judge scores or use oracle labels
   - The reward field must be populated before estimation

3. **"No oracle labels found"**
   - Add oracle labels using `--add-oracle` flag in prepare_cje_data.py
   - Or manually add oracle_label field to some samples