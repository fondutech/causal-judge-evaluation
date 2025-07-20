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
    --output data/arena_prompts.jsonl \
    --max-prompts 1000
```

### 2. Generate Responses
Generate responses using different policies (base, clone, unhelpful):
```bash
python generate_responses.py \
    --prompts data/arena_prompts.jsonl \
    --output-dir data/responses \
    --max-responses 100  # For testing
```

Policies are defined in `policy_config.py`:
- **base**: Standard helpful assistant
- **clone**: Identical to base (control)
- **unhelpful**: Deliberately confusing responses

### 3. Compute Log Probabilities
Compute log P(base_response | prompt) under each policy's model:
```bash
python compute_logprobs.py \
    --responses-dir data/responses \
    --output-dir data/logprobs
```

This computes importance weights for CJE.

### 4. Prepare CJE Dataset
Combine responses and log probabilities into CJE format:
```bash
python prepare_cje_data.py \
    --responses-dir data/responses \
    --logprobs-dir data/logprobs \
    --output data/cje_dataset.jsonl
```

### 5. Add Judge Scores
Score responses using a judge model:
```bash
python add_judge_scores.py --input data/responses/base_responses.jsonl
python add_judge_scores.py --input data/responses/clone_responses.jsonl
python add_judge_scores.py --input data/responses/unhelpful_responses.jsonl
```

Uses Fireworks API with LangChain structured outputs for reliable scoring.
Default model: `llama4-scout-instruct-basic`

### 6. Add Oracle Labels
Add ground truth labels for validation and calibration:
```bash
python add_oracle_labels.py --input data/responses/base_responses.jsonl
python add_oracle_labels.py --input data/responses/clone_responses.jsonl
python add_oracle_labels.py --input data/responses/unhelpful_responses.jsonl
```

Oracle labels are higher-quality evaluations used as ground truth.
Default model: `kimi-k2-instruct`

### 7. Run CJE Analysis
Run the complete analysis with calibration:
```bash
python run_cje_analysis.py \
    --data data/cje_dataset.jsonl \
    --n-folds 5
```

The analysis will use judge scores calibrated to oracle labels. All responses have oracle labels for validation purposes.

## Output

The analysis produces:
- Point estimates for each policy's expected reward
- Standard errors and confidence intervals
- Relative efficiency metrics
- Weight diagnostics

Example output:
```
RESULTS
====================
base     : 0.723 (±0.015, CI: [0.694, 0.752])
clone    : 0.721 (±0.016, CI: [0.690, 0.752])
unhelpful: 0.412 (±0.024, CI: [0.365, 0.459])

🏆 Best policy: base
```

## Key Concepts

1. **Importance Weighting**: We compute log P(base_response | prompt) under each policy to estimate what rewards other policies would have received.

2. **Judge Calibration**: Judge scores are calibrated to oracle labels using isotonic regression with cross-fitting. For ablation studies, you can vary the fraction of oracle labels used for calibration.

3. **Cross-Fitting**: Prevents overfitting in both calibration and importance weight estimation.

## Customization

- **Add new policies**: Edit `policy_config.py`
- **Custom evaluators**: Use `FireworksEvaluator` from `evaluation_utils.py` with custom models/prompts
- **Different datasets**: Modify `prepare_arena_data.py` or create new data preparation scripts

