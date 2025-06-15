<p align="center">
  <img src="docs/img/CJE logo.svg" alt="Causal Judge Evaluation logo"
       width="240" height="auto"/>
</p>

# CJE: Causal Judge Evaluation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-ReadTheDocs-blue.svg)](https://causal-judge-evaluation.readthedocs.io/)

**CJE (Causal Judge Evaluation)** provides unbiased, cost-efficient off-policy evaluation for LLM systems. Compare prompts, models, and parameters using only historical logs—no deployment needed.

### What is a "Policy"?

In CJE, a **policy** is simply a specific LLM configuration—the combination of model, prompt, and parameters that generates responses. For example:
- **Policy A**: GPT-3.5 with a casual prompt and temperature 0.7
- **Policy B**: GPT-4 with a professional prompt and temperature 0.3
- **Policy C**: Claude-3 with a step-by-step reasoning prompt

The **logging policy** is what generated your historical data (past conversations), while **target policies** are the new configurations you want to evaluate. CJE estimates how well target policies would perform without actually deploying them.

## Key Features

- **🎯 Causal, not correlational**: Answers "What would happen if we deployed policy π′?"
- **⚡ Faster evaluation**: Reuses existing responses with teacher-forced scoring
- **📊 Tighter confidence intervals**: Via calibrated doubly-robust estimation
- **🔧 Production-ready**: Caching, checkpointing, multiple provider support
- **📄 Based on research**: Implements the CJE paper (Landesberg 2025) with single-rate efficiency guarantees

> **⚠️ Development Status**: This repository is in active development. APIs may change without notice.

## Installation

```bash
# Clone and install for development
git clone https://github.com/fondutech/causal-judge-evaluation.git
cd causal-judge-evaluation
poetry install

# Set API keys (at least one required)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."  # Optional
export FIREWORKS_API_KEY="..."         # Recommended for teacher forcing
```

📖 See [full installation guide](https://causal-judge-evaluation.readthedocs.io/en/latest/installation.html) for system requirements and troubleshooting.

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

## Experiments

The `experiments/` directory contains standalone research experiments:

- **[Arena 10K Fresh Oracle](experiments/arena_10k_oracle/)** - Validate CJE on 10k ChatBot Arena prompts
  - Human calibration via crowdsourcing
  - Complete workflow with checkpointing
  - Target: High accuracy with significant CI reduction

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