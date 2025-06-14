# Arena 10K Fresh Oracle Experiment

## Overview

The Arena 10K Fresh Oracle experiment is designed to comprehensively validate the Causal Judge Evaluation (CJE) framework using real-world data from ChatBot Arena conversations. This experiment aims to test CJE's ability to accurately predict policy performance with significant efficiency gains.

## Location

All experiment materials are located in:
```
experiments/arena_10k_oracle/
```

## Key Components

### Scripts (in order of execution)

1. **Data Preparation** (`01_prepare_data.py`)
   - Downloads ChatBot Arena conversations from HuggingFace
   - Extracts and samples 10,000 prompts
   - Saves to JSONL format

2. **Logging Policy Generation** (`02_generate_logs.py`)
   - Generates responses using Llama-3-34B-Instruct
   - Computes exact token-level log probabilities
   - Supports checkpointing for large runs

3. **Judge Scoring** (`03_add_judge_scores.py`)
   - Scores responses using same model at T=0
   - Uses 0-10 helpfulness/correctness/safety rubric
   - Produces raw scores for calibration

4. **Human Labeling Export** (`04_export_for_labeling.py`)
   - Exports 2,500 samples for human calibration
   - Formats for Surge AI, MTurk, or generic CSV
   - Includes detailed labeling instructions

5. **Label Import & Calibration** (`04_import_labels.py`)
   - Imports human labels from crowdsourcing
   - Fits isotonic regression for calibration
   - Applies calibration to all 10,000 samples

6. **Target Policy Generation** (`05_generate_targets.py`)
   - Generates responses for alternative policies:
     - π_cot: Chain-of-thought prompting
     - π_rag: Retrieval-augmented generation
     - π_big: Larger 70B model

7. **CJE Estimation** (`06_run_cje.py`)
   - Runs calibrated DR-CPO estimation
   - Computes confidence intervals
   - Generates weight diagnostics

8. **Validation & Analysis** (`07_export_validation.py`, `07_analyze_results.py`)
   - Exports pairs for ground truth comparison
   - Analyzes CJE accuracy vs human evaluation
   - Creates publication-ready figures

### Configuration

The experiment configuration is in:
```yaml
experiments/arena_10k_oracle/configs/arena_10k.yaml
```

Key settings:
- Dataset: ChatBot Arena, 10k samples
- Logging policy: Llama-3-34B, T=0.4
- Judge: Same model, T=0
- Oracle: 25% calibration, 3 votes/sample
- Estimator: 5-fold calibrated DR-CPO

## Running the Experiment

### Phase 1: Initial Data Generation
```bash
cd experiments/arena_10k_oracle/scripts

# Download and prepare data
python 01_prepare_data.py

# Generate logging policy responses
python 02_generate_logs.py

# Add judge scores
python 03_add_judge_scores.py
```

### Phase 2: Human Calibration
```bash
# Export for labeling
python 04_export_for_labeling.py

# >>> Send to crowdsourcing platform <<<
# Collect ~7,500 labels ($600)

# Import and calibrate
python 04_import_labels.py --labels human_labels.csv
```

### Phase 3: Estimation
```bash
# Generate target policies
python 05_generate_targets.py

# Run CJE
python 06_run_cje.py
```

### Phase 4: Validation
```bash
# Export validation pairs
python 07_export_validation.py

# >>> Collect validation labels <<<

# Analyze results
python 07_analyze_results.py
```

## Target Outcomes

Based on the experimental design and paper methodology:

- **Accuracy**: Target ±2pp agreement with ground truth
- **Efficiency**: Target ~70% reduction in confidence interval width
- **Speed**: Target 10× faster than decode+judge baseline
- **Cost**: Target <$1,000 total (including human labels)

*These are expected outcomes. Actual results will be measured and reported after experiment completion.*

## Tips

1. **Use checkpointing** for all generation scripts (auto-enabled for >1000 samples)
2. **Monitor API costs** - each script reports estimated costs
3. **Validate label quality** - check inter-rater agreement in import script
4. **Save intermediate results** - all scripts save progress

## Related Resources

- Main CJE documentation: [docs.cje.ai](https://docs.cje.ai)
- Paper: "Causal Judge Evaluation" (Landesberg 2025)
- Quick test config: `configs/arena_test.yaml` (20 samples)