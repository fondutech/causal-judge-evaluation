# Development Scripts

This directory contains scripts for CJE development workflow.

## Setup

Run once to set up your development environment:

```bash
./scripts/setup-dev.sh
```

This will:
- Configure pip to use standard PyPI (avoiding CodeArtifact issues)
- Install pre-commit hooks
- Set up the development environment

## Daily Development

### Code Quality

```bash
# Check code formatting and types
./scripts/lint.sh

# Auto-fix code formatting
./scripts/format.sh

# Run pre-commit hooks manually
./scripts/pre-commit-run.sh run --all-files
```

### Testing

```bash
# Run our pipeline integration tests
python -m pytest tests/test_pipeline_integration.py -v

# Run all tests
python -m pytest tests/ -v
```

## Pre-commit Hooks

Pre-commit hooks are automatically installed and will run on every commit. They include:

- **Black**: Code formatting
- **MyPy**: Type checking

If you encounter issues with pre-commit hooks due to custom PyPI indexes, use the wrapper script:

```bash
./scripts/pre-commit-run.sh run --files <file1> <file2>
```

## Troubleshooting

### Pre-commit Issues with Custom PyPI

If you see errors related to AWS CodeArtifact or custom PyPI indexes:

1. Use the wrapper script: `./scripts/pre-commit-run.sh`
2. Or run the setup script again: `./scripts/setup-dev.sh`
3. As a fallback, use the direct tools: `./scripts/lint.sh` and `./scripts/format.sh`

# Arena Analysis Scripts

This directory contains command-line scripts for running CJE experiments on Chatbot Arena data.

## Quick Start

```bash
# Set your API key
export OPENAI_API_KEY="sk-your-key-here"

# Run with default settings (1000 samples, 5 models)
python scripts/run_arena_analysis.py

# Quick test run (100 samples, 2 models)
python scripts/run_arena_analysis.py --max-samples 100 --target-models gpt-3.5-turbo gpt-4
```

## 📖 Full Documentation

For comprehensive usage instructions, configuration options, troubleshooting, and examples, see:

**[Arena Runner Guide](../docs/arena_runner_guide.md)**

## Scripts

- **`run_arena_analysis.py`** - Main one-command runner for token-level CJE analysis
- `arena_workflow.py` - Legacy workflow script  
- `arena_diagnostics.py` - Diagnostic and validation utilities
- `make_figs.py` - Figure generation utilities
- `ci_summary.py` - Confidence interval summary tools

## Output

Results are saved to timestamped directories in `outputs/arena_runs/` with:
- JSON results with confidence intervals
- Calibration and comparison plots  
- Raw interaction logs (JSONL format)
- ZIP bundle for easy sharing 