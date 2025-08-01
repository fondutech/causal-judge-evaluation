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
export OPENAI_API_KEY="your-key-here"  # For judge and oracle models
export FIREWORKS_API_KEY="your-key-here"  # For response generation and log probs

# Install dependencies
cd /path/to/cje_simplified
poetry install
```

## Directory Structure

```
arena_10k_simplified/
├── pipeline_steps/              # Data preparation modules
│   ├── prepare_arena_data.py    # Extract prompts from ChatBot Arena
│   ├── generate_responses.py    # Generate responses for each policy
│   ├── add_judge_scores.py      # Add lightweight evaluations
│   ├── add_oracle_labels.py     # Add high-quality evaluations
│   ├── compute_logprobs.py      # Compute log probabilities
│   └── prepare_cje_data.py      # Combine into CJE format
│
├── test_pipeline.py             # Test pipeline (50 samples)
├── generate_arena_data.py       # Production pipeline (1000+ samples)
├── analyze_dataset.py           # Run CJE estimation
├── analyze_oracle_coverage.py   # Ablation studies
│
├── test_e2e_data/               # Test outputs (50 samples)
└── data/                        # Production outputs (1000+ samples)
    ├── prompts.jsonl
    ├── responses/
    ├── logprobs/
    └── cje_dataset.jsonl
```

## Quick Start

### Testing Pipeline (50 samples)
Run a quick test to verify everything works:
```bash
python test_pipeline.py --n-samples 50
```

This runs the complete pipeline on 50 samples for testing.

### Production Pipeline (1000+ samples)
Run the full production data preparation:
```bash
python generate_arena_data.py \
    --n-samples 1000 \
    --max-tokens 256 \
    --oracle-coverage 0.5
```

This runs all pipeline steps automatically and saves to `data/`.

### Incremental Runs
To skip existing files and only run missing steps:
```bash
python generate_arena_data.py \
    --n-samples 1000 \
    --skip-existing
```

## Detailed Pipeline Steps

The pipeline consists of 6 data preparation steps (in `pipeline_steps/` directory):

### 1. Extract Arena Prompts
```bash
python pipeline_steps/prepare_arena_data.py \
    --samples 1000 \
    --output data/prompts.jsonl
```
- Downloads ChatBot Arena dataset from HuggingFace
- Filters for English conversations using language field
- Extracts unique first-turn user prompts
- Avoids duplicates to ensure fresh responses

### 2. Generate Responses
```bash
python pipeline_steps/generate_responses.py \
    --prompts data/prompts.jsonl \
    --output-dir data/responses \
    --max-tokens 256
```
Policies (defined in `policy_config.py`):
- **base**: Standard helpful assistant
- **clone**: Identical to base (control)
- **unhelpful**: Deliberately confusing responses

### 3. Add Judge Scores
```bash
python pipeline_steps/add_judge_scores.py \
    --input data/responses/base_responses.jsonl
```
- Uses OpenAI API with structured outputs
- Default model: `gpt-4.1-nano-2025-04-14`
- Adds scores to response files in-place

### 4. Add Oracle Labels
```bash
python pipeline_steps/add_oracle_labels.py \
    --input data/responses/base_responses.jsonl
```
- Higher-quality evaluations for ground truth
- Default model: `o4-mini-2025-04-16`
- All responses get oracle labels for validation

### 5. Compute Log Probabilities
```bash
python pipeline_steps/compute_logprobs.py \
    --responses-dir data/responses \
    --output-dir data/logprobs
```
- Computes log P(base_response | prompt) under each policy
- Uses median of 3 samples to handle API variance
- Required for importance weight calculation

### 6. Prepare CJE Dataset
```bash
python pipeline_steps/prepare_cje_data.py \
    --responses-dir data/responses \
    --logprobs-dir data/logprobs \
    --output data/cje_dataset.jsonl \
    --oracle-coverage 0.5
```
- Combines all data into CJE format
- Calibrates judge scores using oracle labels
- Oracle coverage controls calibration subset

## Analysis and Ablation Studies

### Run CJE Analysis
Analyze a prepared dataset:
```bash
python analyze_dataset.py \
    --data data/cje_dataset.jsonl \
    --n-folds 5 \
    --output data/results.json \
    --plot-dir data/plots
```

### Oracle Coverage Ablation
Study the effect of oracle coverage on estimation quality:
```bash
# Prepare datasets with different oracle coverage
python create_oracle_coverage_variants.py \
    --response-dir data/responses \
    --logprob-dir data/logprobs \
    --oracle-fraction 0.25 \
    --output data/ablation_data/oracle_0_25.jsonl

# Run ablation study
python analyze_oracle_coverage.py \
    --data-dir data/ablation_data \
    --output-dir data/ablation_results
```

This compares:
- Different oracle coverage levels (25%, 50%, 100%)
- Different estimators (CalibratedIPS, RawIPS)
- Multiple random seeds for robustness

**Options:**
- `--use-oracle`: Use oracle labels directly as rewards (skip calibration)
- `--output`: Save results to a JSON file (optional)
- `--n-folds`: Number of cross-fitting folds for importance weight calibration (default: 5, minimum: 2)
- `--judge-field`: Field containing judge scores (default: "judge_score")
- `--oracle-field`: Field containing oracle labels (default: "oracle_label")
- `--plot-dir`: Directory to save visualization plots (requires matplotlib)
- `--no-plots`: Disable plot generation even if matplotlib is available

Note: The `--n-folds` parameter is used for calibrating importance weights to ensure unbiased estimation, even when rewards are pre-computed. This is separate from any judge score calibration.

**Visualizations:**
When `--plot-dir` is specified, the analysis generates:
- **Weight distributions**: Histograms showing importance weight distributions for each policy
- **ESS comparison**: Bar chart comparing Effective Sample Size across policies
- **Weight summary**: Combined visualization of weight statistics and diagnostics
- **Calibration comparison**: Reliability diagram comparing judge vs calibrated scores (if calibration was performed)

## Running Ablation Studies

The ablation study framework allows systematic comparison of different estimators and oracle coverage levels.

### Quick Start: Run Complete Ablation Study

```bash
# Run all experiments (oracle coverage: 25%, 50%, 100% × estimators: CalibratedIPS, RawIPS)
python analyze_oracle_coverage.py

# The script will:
# 1. Use existing ablation datasets (or create them if --prepare-data is used)
# 2. Run each estimator on each dataset
# 3. Generate a summary table and visualizations
```

### Step-by-Step Workflow

#### 1. Prepare Ablation Datasets

Create datasets with different oracle coverage levels:

```bash
# 25% oracle coverage
python create_oracle_coverage_variants.py \
    --oracle-fraction 0.25 \
    --seed 42 \
    --output test_e2e_data/ablation_data/oracle_0_25_seed42.jsonl

# 50% oracle coverage  
python create_oracle_coverage_variants.py \
    --oracle-fraction 0.50 \
    --seed 42 \
    --output test_e2e_data/ablation_data/oracle_0_50_seed42.jsonl

# 100% oracle coverage
python create_oracle_coverage_variants.py \
    --oracle-fraction 1.00 \
    --seed 42 \
    --output test_e2e_data/ablation_data/oracle_1_00_seed42.jsonl
```

This script:
- Reads immutable response and logprob files
- Randomly selects the specified fraction of samples for oracle calibration
- Uses those samples to calibrate judge scores for ALL samples
- Creates a CJE-ready dataset with calibrated rewards

#### 2. Run Individual Experiments

Test different estimators on each dataset:

```bash
# CalibratedIPS (with isotonic weight calibration)
python analyze_dataset.py \
    --data test_e2e_data/ablation_data/oracle_0_25_seed42.jsonl \
    --estimator calibrated-ips \
    --output results/calibrated_ips_25.json

# RawIPS (standard importance sampling)
python analyze_dataset.py \
    --data test_e2e_data/ablation_data/oracle_0_25_seed42.jsonl \
    --estimator raw-ips \
    --output results/raw_ips_25.json
```

#### 3. Analyze Results

The ablation runner generates:
- `ablation_summary.csv` - All metrics in tabular format
- Console output with formatted comparisons
- Policy-specific estimates and standard errors
- Effective Sample Size (ESS) comparisons

### Available Estimators

- **calibrated-ips**: Calibrated importance sampling with isotonic regression on weights
- **raw-ips**: Standard importance sampling without weight calibration

### Key Findings from Ablations

1. **Weight calibration dramatically reduces variance** for policies with extreme weights
2. **25% oracle coverage is often sufficient** - more oracle data shows diminishing returns
3. **The unhelpful policy** creates extreme weights (up to 100 after clipping) that challenge standard IPS

### Extending the Framework

To add new estimators:
1. Create a new estimator class inheriting from `BaseCJEEstimator`
2. Add it to `analyze_dataset.py`'s estimator choices
3. Update `analyze_oracle_coverage.py`'s ABLATION_CONFIG

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
python test_pipeline.py
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
python test_pipeline.py --no-cleanup --test-dir my_test_data
```

## Customization

- **Add new policies**: Edit `policy_config.py`
- **Custom evaluators**: Use `FireworksEvaluator` from `evaluation_utils.py` with custom models/prompts
- **Different datasets**: Modify `prepare_arena_data.py` or create new data preparation scripts

