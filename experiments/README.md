# CJE Experiments

This directory contains standalone experiments that demonstrate and validate the Causal Judge Evaluation (CJE) framework.

## Available Experiments

### 1. Arena 10K Oracle Experiment (`arena_10k_oracle/`)

A comprehensive experiment demonstrating CJE on 10,000 ChatBot Arena prompts using deterministic llama.cpp teacher forcing.

**Key Features:**
- Real-world data from ChatBot Arena conversations (English-only)
- Deterministic teacher forcing with llama.cpp (no API non-determinism!)
- 2 target policies: pi_clone (validation), pi_bad (deliberately unhelpful)
- Ground truth validation via GPT-4o oracle labels
- Complete weight diagnostics and ESS analysis

**Unique Advantages:**
- 100% reproducible log probabilities
- No API costs for teacher forcing
- Pi_clone weights exactly 1.0 (perfect validation)
- GPU accelerated (Metal/CUDA support)

**Requirements:**
- llama-cpp-python installed
- Llama 3.2 3B model (~2.5GB download)
- OpenAI API key for judge/oracle only

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