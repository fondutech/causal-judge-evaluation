# Causal Judge Evaluation (CJE)

Fast, accurate model evaluation using AI judges with causal inference.

## Overview

CJE solves the problem of evaluating LLM improvements by:
- Using AI judges instead of expensive human evaluation
- Applying causal inference to correct for judge biases
- Achieving 70%+ variance reduction and 10× GPU efficiency
- Providing calibrated uncertainty estimates

## Installation

```bash
# Clone the repository
git clone https://github.com/yolandelandsberg/causal-judge-evaluation.git
cd causal-judge-evaluation

# Install with Poetry (recommended)
make dev-setup

# Or install directly
poetry install
```

## Quick Start

```python
from cje import run_experiment

# Run evaluation with example config
results = run_experiment("configs/example_eval.yaml")
print(f"Policy improvement: {results['improvement']:.3f} ± {results['ci_width']:.3f}")
```

## Key Features

- **Multiple Estimators**: IPS, SNIPS, CalibratedIPS, DRCPO, MRDR
- **Uncertainty Quantification**: Built-in support for judge uncertainty
- **Provider Support**: OpenAI, Anthropic, Fireworks, Together, Google
- **Efficient**: Teacher forcing for unbiased log probabilities
- **Robust**: Automatic calibration and cross-fitting

## Documentation

- [Full Documentation](https://causal-judge-evaluation.readthedocs.io)
- [API Reference](https://causal-judge-evaluation.readthedocs.io/api)
- [Arena 10K Experiment](experiments/arena_10k_oracle/README.md)

## Development

```bash
# Run tests
make test

# Run linting (required before commits)
make lint

# Build documentation
make docs
```

## License

MIT License - see LICENSE file for details.