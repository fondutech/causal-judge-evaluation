# CJE Experiments

This directory contains standalone experiments that demonstrate and validate the Causal Judge Evaluation (CJE) framework.

## Available Experiments

### 1. Arena 10K Fresh Oracle Experiment (`arena_10k_oracle/`)

A comprehensive experiment demonstrating CJE on 10,000 ChatBot Arena prompts with fresh human oracle labels.

**Key Features:**
- Real-world data from ChatBot Arena conversations
- Human calibration via crowdsourcing (2,500 samples)
- Multiple target policies (CoT, RAG, 70B model)
- Ground truth validation
- Complete cost and efficiency analysis

**Expected Outcomes:**
- Accuracy: Target ±2pp vs ground truth
- Efficiency: ~70% CI reduction, 10× GPU speedup  
- Cost: <$1,000 total (including human labels)

*Note: This is a planned experiment. Actual results will be reported after completion.*

See [arena_10k_oracle/README.md](arena_10k_oracle/README.md) for detailed instructions.

## Running Experiments

Each experiment is self-contained with its own:
- Scripts for each step
- Configuration files
- Data directory
- Documentation

General workflow:
1. Navigate to the experiment directory
2. Follow the README instructions
3. Run scripts in order (some have natural breakpoints for human labeling)

## Creating New Experiments

To create a new experiment:

1. Create a new directory: `experiments/your_experiment_name/`
2. Add subdirectories:
   ```
   your_experiment_name/
   ├── README.md          # Detailed instructions
   ├── configs/           # Experiment configurations
   ├── scripts/           # Step-by-step scripts
   ├── data/              # Data storage
   └── outputs/           # Results and figures
   ```
3. Follow the pattern from `arena_10k_oracle/` for script organization
4. Document clearly, especially any manual steps (e.g., human labeling)

## Integration with Main Codebase

These experiments use the core CJE library but are designed to be:
- **Standalone**: Can be run independently
- **Reproducible**: Fixed seeds and versioned data
- **Educational**: Show complete workflows with real data

For quick testing, use the configs in the main `configs/` directory:
- `configs/arena_test.yaml` - Quick 20-sample test
- `configs/arena_10k_oracle.yaml` - Full 10k research run