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

 