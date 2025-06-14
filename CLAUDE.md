# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## IMPORTANT: Code Quality Requirements

**BEFORE COMPLETING ANY TASK**, you MUST run the following commands to ensure code quality:
```bash
make lint        # Runs both black formatting and mypy type checking
```

Both black and mypy MUST pass without errors before considering any task complete. If there are errors:
1. Fix all black formatting issues (usually automatic)
2. Fix all mypy type errors
3. Run `make lint` again to verify all issues are resolved

## Essential Commands

### Development Setup
```bash
make dev-setup    # Install dependencies with Poetry and set up pre-commit hooks
```

### Running Tests
```bash
# Basic test run
poetry run pytest

# Include slow tests
poetry run pytest --run-slow

# Run specific test categories
poetry run pytest --theoretical-only    # Theoretical guarantees only
poetry run pytest --empirical-only      # Empirical validation only
poetry run pytest --paper-validation    # Paper claim validation

# Run a single test
poetry run pytest tests/test_file.py::test_function_name
```

### Linting and Formatting
```bash
make lint        # Run both black formatting and mypy type checking
make format      # Format code with black only
make mypy        # Type check only

# Or directly:
black cje/ examples/ tests/
mypy cje/ examples/ --exclude ".*test.*" --ignore-missing-imports --no-strict-optional
```

### Documentation
```bash
make docs        # Build documentation
make docs-serve  # Serve documentation locally at http://localhost:8000
make docs-clean  # Clean rebuild documentation
```

### Running CJE Experiments
```bash
# Full experiment pipeline
cje run --cfg-path configs --cfg-name arena_test

# Individual pipeline stages
cje log           # Compute log probabilities
cje judge         # Generate judge scores
cje calibrate     # Calibrate judge
cje estimate      # Run causal estimation

# Arena experiment
python examples/run_arena_experiment.py --config arena_experiment
```

## High-Level Architecture

### Core Pipeline Flow
The CJE pipeline follows this sequence:
1. **Data Loading** (`cje/data/`) - Loads datasets (ChatbotArena, CSV, JSONL)
2. **Log Probability Computation** (`cje/loggers/`) - Computes policy log probabilities
3. **Judge Scoring** (`cje/judge/`) - Scores responses using LLM judges
4. **Calibration** (`cje/calibration/`) - Isotonic calibration with cross-fitting
5. **Causal Estimation** (`cje/estimators/`) - DRCPO/MRDR/IPS estimation
6. **Results** (`cje/results/`) - Policy rankings and diagnostics

### Key Design Patterns

**Provider Abstraction**: All LLM interactions go through provider interfaces in `cje/judge/providers/`. When adding new LLM providers, implement both regular and structured variants.

**Configuration System**: Uses Hydra with unified config schema in `cje/config/unified.py`. All experiments are driven by YAML configs in `configs/`.

**Caching Strategy**: Extensive caching for expensive operations:
- LLM API calls cached in `cje/utils/inference_cache.py`
- Pipeline results cached based on work_dir in config
- Checkpointing support for long-running experiments

**Weight Diagnostics**: Built-in importance weight quality checks in `cje/utils/weight_diagnostics.py`. Always monitor ESS (Effective Sample Size) warnings.

### Research Components

The `cje/research/` module contains experimental features:
- `arena_experiment.py` - Full arena-style evaluation orchestrator
- `phase_manager.py` - Manages multi-phase research experiments
- `validation.py` - Gold standard validation runners

### Critical Implementation Details

**Cross-Fitting**: The k-fold cross-validation in calibration prevents overfitting. Default k=5, but can be adjusted based on data size.

**Log Ratio Clipping**: Default clip value of 20.0 prevents extreme importance weights. This is configurable in diagnostics section.

**Oracle Mode**: When enabled, uses a stronger model for ground truth labels. Useful for validation but increases API costs.

**Trajectory Support**: The codebase supports both single-turn and multi-turn trajectory evaluation via `cje/data/trajectory_dataset.py` and `cje/estimators/trajectory_drcpo.py`.

**Weight Calibration Bug Fix**: Fixed critical bug where re-scaling after capping could push weights above the cap again. The implementation now accepts small bias (E[w] ≠ 1) to maintain variance control, which is preferable to violating the cap constraint. See `cje/estimators/calibration.py` for the corrected implementation.

### Paper Implementation Notes

This codebase implements the CJE paper (Landesberg 2025) with extensions:

**Core Paper Components**:
- Calibrated DR-CPO estimator (Section 4.2) → `cje/estimators/drcpo.py`
- Isotonic calibration (Section 2.2) → `cje/calibratrunion/isotonic.py`
- MRDR variant (Section 4.3) → `cje/estimators/mrdr.py`
- Cross-fitted algorithm (Algorithm 1) → Implemented in all estimators
- Single-rate efficiency (Theorem 5.2) → Preserved in implementation

**Key Extensions**:
- Multi-policy vectorized evaluation (paper focuses on single π')
- Comprehensive provider support beyond paper examples
- Arena research framework in `cje/research/`
- Production features: caching, checkpointing, progress tracking

**Recent Development**:
- Pi0 data generation scripts in `scripts/generate_pi0_data.py`
- Arena research experiments in `configs/arena_research_experiment.yaml`
- Validation framework for gold standard comparisons
- Fixed critical weight calibration bug (no re-scaling after capping)
- Implemented base cross-fitted estimator in `cje/estimators/base_crossfit.py`
- Added comprehensive visualization utilities in `cje/results/visualization.py`
- Added missing weight stabilization methods (SWITCH, log-exp)

### Documentation Notes

**Package Status**: CJE is not yet published to PyPI. All installation must be done via development setup with Poetry.

**Config References**: Use `arena_test` as the default config example for quick testing (20 samples), or `arena_research_experiment` for full runs (10k samples).

**Common Documentation Issues**:
- The `cje results` command does not exist - results are accessed via Python API or saved JSON files
- Installation should always use `poetry install` after cloning, never `pip install cje`
- When updating docs, ensure ReadTheDocs rebuilds by pushing to GitHub

**Pre-commit Hooks**: The repository uses strict pre-commit hooks including:
- Black formatting (automatic fixes)
- Mypy type checking (may block commits)
- To bypass temporarily: `git commit --no-verify`

**Type Annotations**: When adding new code:
- Always add type hints to function signatures
- Use `Optional[T]` instead of `T = None` for optional types
- Import types from `typing` module
- For numpy arrays in matplotlib, use `.tolist()` to avoid mypy issues