<p align="center">
  <img src="docs/img/CJE logo.svg" alt="Causal Judge Evaluation logo"
       width="240" height="auto"/>
</p>

# CJE: Causal Judge Evaluation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-ReadTheDocs-blue.svg)](https://causal-judge-evaluation.readthedocs.io/)

**CJE** is a toolkit for causally evaluating Large Language Model (LLM) policies using logged interaction data. It provides unbiased counterfactual metrics for comparing prompts, models, and parameters without deployment.

> **⚠️ Development Status**: This repository is in active development. APIs may change without notice.

## Installation

```bash
# Clone and install for development
git clone https://github.com/fondutech/causal-judge-evaluation.git
cd causal-judge-evaluation
poetry install

# Set API keys
export OPENAI_API_KEY="sk-..."
```

## Quick Start

```bash
# Run evaluation on sample data
cje run --cfg-path configs --cfg-name arena_test

# Or use Python API
from cje.pipeline import run_pipeline
results = run_pipeline(cfg_path="configs", cfg_name="arena_test")
```

## Documentation

**📖 [Full Documentation](https://causal-judge-evaluation.readthedocs.io/)** - Comprehensive guides, API reference, and tutorials

- [Installation Guide](https://causal-judge-evaluation.readthedocs.io/en/latest/installation.html)
- [5-Minute Quickstart](https://causal-judge-evaluation.readthedocs.io/en/latest/quickstart.html)
- [User Guide](https://causal-judge-evaluation.readthedocs.io/en/latest/guides/user_guide.html)
- [API Reference](https://causal-judge-evaluation.readthedocs.io/en/latest/api/index.html)

## Development

```bash
# Run tests
make test

# Format code
make format

# Build docs locally
make docs
```

## License

MIT License. See [LICENSE](LICENSE) for details.