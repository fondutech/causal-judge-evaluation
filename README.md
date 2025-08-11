# CJE Simplified

Production-ready implementation of Causal Judge Evaluation for unbiased LLM evaluation.

## Overview

CJE Simplified provides a clean, minimal implementation of the CJE methodology:
- **Unbiased Estimation**: Corrects for distribution shift between policies
- **Variance Control**: Isotonic calibration prevents weight explosion  
- **Doubly Robust**: Optional DR estimation for better bias-variance tradeoff
- **Production Ready**: Clean API, comprehensive tests, type hints throughout

## Installation

```bash
git clone https://github.com/fondutech/causal-judge-evaluation.git
cd causal-judge-evaluation
pip install -e .
```

## Quick Start

```python
from cje_simplified import (
    load_dataset_from_jsonl,
    PrecomputedSampler,
    CalibratedIPS
)

# Load data with precomputed log probabilities
dataset = load_dataset_from_jsonl("data.jsonl")

# Create sampler and estimator
sampler = PrecomputedSampler(dataset)
estimator = CalibratedIPS(sampler)

# Get unbiased policy estimates
results = estimator.fit_and_estimate()
print(f"Best policy: {results.best_policy()}")
```

## Key Features

### Estimators
- **CalibratedIPS** - Variance-controlled IPS with isotonic calibration (recommended)
- **RawIPS** - Standard importance sampling with clipping
- **DRCPOEstimator** - Doubly robust with cross-fitted outcome models

### Data Format
```json
{
  "prompt": "What is machine learning?",
  "response": "Machine learning is...",
  "base_policy_logprob": -35.704,
  "target_policy_logprobs": {
    "gpt4": -32.456,
    "claude": -33.789
  },
  "metadata": {
    "judge_score": 0.85,
    "oracle_label": 0.90
  }
}
```

### Computing Log Probabilities
```python
from cje_simplified import compute_teacher_forced_logprob

result = compute_teacher_forced_logprob(
    prompt="What is 2+2?",
    response="The answer is 4.",
    model="accounts/fireworks/models/llama-v3p2-3b-instruct"
)
```

## Documentation

Full documentation available at: https://causal-judge-evaluation.readthedocs.io

- [Getting Started](docs/getting_started.rst)
- [Data Format Guide](docs/data_format.rst)
- [Estimators Guide](docs/estimators.rst)
- [API Reference](docs/api/)

## Testing

```bash
# Run all tests
poetry run pytest cje_simplified/tests/

# Run specific test
poetry run pytest cje_simplified/tests/test_pipeline.py -v
```

## License

MIT License - see LICENSE file for details.