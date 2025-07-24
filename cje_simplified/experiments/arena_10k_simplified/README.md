# Arena 10K Simplified Experiment

This experiment demonstrates the complete CJE (Causal Judge Evaluation) pipeline using the ChatGPT Arena dataset.

## Pipeline Overview

The CJE pipeline evaluates different LLM policies by:
1. Generating responses from a base policy
2. Computing importance weights using log probabilities
3. Calibrating judge scores to ground truth labels
4. Estimating counterfactual rewards for each policy

## Prerequisites

```bash
# Set API keys
export FIREWORKS_API_KEY="your-key-here"

# Install dependencies
cd /path/to/cje_simplified
poetry install
```

## Step-by-Step Pipeline

### 1. Prepare Arena Data
Extract prompts from the ChatGPT Arena dataset:
```bash
python prepare_arena_data.py \
    --input /path/to/chatbot_arena_conversations.json \
    --output data/prompts.jsonl \
    --max-prompts 1000
```

Extracts the first user turn from Arena conversations.

### 2. Generate Responses
Generate responses using different policies (base, clone, unhelpful):
```bash
python generate_responses.py \
    --prompts data/prompts.jsonl \
    --output-dir data/responses \
    --max-responses 100  # For testing
```

Policies are defined in `policy_config.py`:
- **base**: Standard helpful assistant
- **clone**: Identical to base (control)
- **unhelpful**: Deliberately confusing responses

### 3. Add Judge Scores
Score responses using a judge model:
```bash
python add_judge_scores.py --input data/responses/base_responses.jsonl
python add_judge_scores.py --input data/responses/clone_responses.jsonl
python add_judge_scores.py --input data/responses/unhelpful_responses.jsonl
```

Uses Fireworks API with LangChain structured outputs for reliable scoring.
Default model: `llama4-scout-instruct-basic`

### 4. Add Oracle Labels
Add ground truth labels for validation and calibration:
```bash
python add_oracle_labels.py --input data/responses/base_responses.jsonl
python add_oracle_labels.py --input data/responses/clone_responses.jsonl
python add_oracle_labels.py --input data/responses/unhelpful_responses.jsonl
```

Oracle labels are higher-quality evaluations used as ground truth.
Default model: `kimi-k2-instruct`
All responses get oracle labels for validation purposes.

### 5. Compute Log Probabilities
Compute log P(base_response | prompt) under each policy's model:
```bash
python compute_logprobs.py \
    --responses-dir data/responses \
    --output-dir data/logprobs
```

This computes importance weights for CJE. Uses median of 3 samples to handle API variance.

### 6. Prepare CJE Dataset
Combine responses and log probabilities into CJE format:
```bash
python prepare_cje_data.py \
    --responses-dir data/responses \
    --logprobs-dir data/logprobs \
    --output data/cje_dataset.jsonl
```

This script reads judge scores and oracle labels from the response files.

**Advanced: Pre-compute rewards with different oracle coverage**
```bash
# 100% oracle coverage - use oracle labels directly as rewards
python prepare_cje_data.py \
    --responses-dir data/responses \
    --logprobs-dir data/logprobs \
    --output data/cje_dataset_oracle_100.jsonl \
    --oracle-coverage 1.0

# 50% oracle coverage - calibrate judge scores using 50% of oracle labels
python prepare_cje_data.py \
    --responses-dir data/responses \
    --logprobs-dir data/logprobs \
    --output data/cje_dataset_calibrated_50.jsonl \
    --oracle-coverage 0.5 \
    --seed 42
```

### 7. Run CJE Analysis
Run the complete analysis:
```bash
python run_cje_analysis.py \
    --data data/cje_dataset.jsonl \
    --n-folds 5 \
    --output results.json
```

When rewards are pre-computed in the dataset (using `--oracle-coverage` in step 6), the analysis will use those rewards directly. Otherwise, it will calibrate judge scores to oracle labels.

**Options:**
- `--use-oracle`: Use oracle labels directly as rewards (skip calibration)
- `--output`: Save results to a JSON file (optional)
- `--n-folds`: Number of cross-fitting folds for importance weight calibration (default: 5, minimum: 2)
- `--judge-field`: Field containing judge scores (default: "judge_score")
- `--oracle-field`: Field containing oracle labels (default: "oracle_label")

Note: The `--n-folds` parameter is used for calibrating importance weights to ensure unbiased estimation, even when rewards are pre-computed. This is separate from any judge score calibration.

## Running Ablation Studies

To compare different oracle coverage levels:

```bash
# 1. Create dataset with 100% oracle coverage (oracle labels as rewards)
python prepare_cje_data.py \
    --responses-dir data/responses \
    --logprobs-dir data/logprobs \
    --output data/cje_oracle_100.jsonl \
    --oracle-coverage 1.0

# 2. Create dataset with 50% oracle coverage (calibrated judge scores)
python prepare_cje_data.py \
    --responses-dir data/responses \
    --logprobs-dir data/logprobs \
    --output data/cje_calibrated_50.jsonl \
    --oracle-coverage 0.5 \
    --seed 42

# 3. Run analysis on both datasets
python run_cje_analysis.py --data data/cje_oracle_100.jsonl --output results_oracle_100.json
python run_cje_analysis.py --data data/cje_calibrated_50.jsonl --output results_calibrated_50.json

# 4. Compare results
diff results_oracle_100.json results_calibrated_50.json
```

The results will show how calibration quality affects CJE estimates.

## Output

The analysis produces:
- Point estimates for each policy's expected reward
- Standard errors and confidence intervals
- Relative efficiency metrics
- Weight diagnostics

Example console output:
```
4. Results:
   ----------------------------------------
   base (observed):
     Estimate: 0.787
     Std Error: 0.081
     95% CI: [0.628, 0.946]
   clone:
     Estimate: 0.721
     Std Error: 0.016
     95% CI: [0.690, 0.752]
   unhelpful:
     Estimate: 0.412
     Std Error: 0.024
     95% CI: [0.365, 0.459]

   🏆 Best policy: base
```

Example JSON output (when using `--output`):
```json
{
  "timestamp": "2024-01-20T10:15:30.123456",
  "dataset": {
    "path": "data/cje_dataset.jsonl",
    "n_samples": 100,
    "target_policies": ["clone", "unhelpful"]
  },
  "workflow": "judge_calibration",
  "estimation": {
    "n_folds": 5,
    "policies": {
      "base": {
        "estimate": 0.787,
        "standard_error": 0.081,
        "ci_lower": 0.628,
        "ci_upper": 0.946,
        "type": "observed",
        "n_samples": 100
      },
      "clone": {
        "estimate": 0.721,
        "standard_error": 0.016,
        "ci_lower": 0.690,
        "ci_upper": 0.752,
        "type": "counterfactual"
      },
      "unhelpful": {
        "estimate": 0.412,
        "standard_error": 0.024,
        "ci_lower": 0.365,
        "ci_upper": 0.459,
        "type": "counterfactual"
      }
    }
  },
  "best_policy": "base",
  "weight_diagnostics": {
    "mean_weight": 1.02,
    "max_weight": 3.45,
    "effective_sample_size": 87.3,
    "effective_sample_size_fraction": 0.873
  },
  "calibration": {
    "n_oracle": 100,
    "calibration_rmse": 0.082,
    "coverage_at_01": 0.856
  }
}
```

## Key Concepts

1. **Importance Weighting**: We compute log P(base_response | prompt) under each policy to estimate what rewards other policies would have received. This means all policies are evaluated on the **same base policy responses**, weighted by how likely each policy is to generate those responses.

2. **Judge Calibration**: Judge scores are calibrated to oracle labels using isotonic regression with cross-fitting. For ablation studies, you can vary the fraction of oracle labels used for calibration.

3. **Cross-Fitting**: Prevents overfitting in both calibration and importance weight estimation.

4. **Median Log Probabilities**: To handle API non-determinism, we compute log probabilities 3 times per prompt/policy and use the median value. This prevents outlier values from distorting importance weights.

## Understanding CJE Results

When running CJE analysis, you'll see results for:
- **Base policy (observed)**: The average reward of the actual responses in your dataset
- **Target policies (counterfactual)**: Estimated rewards if those policies were deployed

All policies may show similar estimates when evaluated on high-quality base responses, as the importance weighting compensates for probability differences. To see policy differences, you need sufficient variation in response quality or policies with very different response distributions.

## Data Flow

1. Response files are modified in place to add judge scores and oracle labels
2. Log probabilities are computed for BASE responses under all policy models
3. The final CJE dataset combines data from both response and logprob files

## Testing

Run the end-to-end test to verify the pipeline:
```bash
python test_e2e_pipeline.py
```

This test:
- Runs the complete pipeline on 10 test prompts
- Verifies each step produces expected outputs
- Tests all three policies (base, clone, unhelpful)
- Validates data format and integrity
- Runs full CJE analysis with calibration
- Saves results to `test_e2e_data/cje_results.json`
- Displays results summary including base policy
- Cleans up test files automatically (use `--no-cleanup` to preserve)

For debugging:
```bash
python test_e2e_pipeline.py --no-cleanup --test-dir my_test_data
```

## Customization

- **Add new policies**: Edit `policy_config.py`
- **Custom evaluators**: Use `FireworksEvaluator` from `evaluation_utils.py` with custom models/prompts
- **Different datasets**: Modify `prepare_arena_data.py` or create new data preparation scripts

