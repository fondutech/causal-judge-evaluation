<p align="center">
  <img src="docs/img/CJE logo.svg" alt="Causal Judge Evaluation logo"
       width="240" height="auto"/>
</p>

# CJE-Core: Causal Judge Evaluation Toolkit

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-ReadTheDocs-blue.svg)](https://causal-judge-evaluation.readthedocs.io/)
![Development Status](https://img.shields.io/badge/status-active%20development-orange.svg)
![Production Ready](https://img.shields.io/badge/production%20ready-no-red.svg)

> **⚠️ DEVELOPMENT STATUS WARNING**
> 
> **This repository is in active development and is NOT ready for production use.**
> 
> - 🚧 **APIs may change without notice**
> - 🔬 **Suitable for research, experimentation, and evaluation**  
> - 🚫 **Do not use in production systems or critical applications**
> - 📝 **Documentation and examples are provided for educational purposes**
> 
> We welcome early feedback and contributions, but please be aware that breaking changes may occur frequently as we work toward a stable release.

**CJE** (Causal Judge Evaluation) is a comprehensive toolkit for **Causal Judge Evaluation** of Large Language Models (LLMs), providing robust off-policy evaluation methods for comparing language model policies using logged data.

## 🎯 What CJE Does

- **Estimate Policy Improvements** without deployment using doubly-robust causal inference
- **Compare LLM Policies** (prompts, models, parameters) using logged interaction data  
- **Diagnose Evaluation Quality** with built-in reliability assessment and weight analysis
- **Scale Production Workloads** with robust error handling, caching, and progress tracking

## 📚 Documentation Hub

**📖 [Complete Documentation](https://causal-judge-evaluation.readthedocs.io/)** - Professional docs with search and PDF export

### Quick Navigation by User Type

**🚀 New Users**
- [5-Minute Quickstart](https://causal-judge-evaluation.readthedocs.io/en/latest/quickstart.html) - Get running immediately
- [Installation Guide](https://causal-judge-evaluation.readthedocs.io/en/latest/installation.html) - Setup and requirements

**⚙️ Practitioners**  
- [User Guide](https://causal-judge-evaluation.readthedocs.io/en/latest/guides/user_guide.html) - Real-world workflows and troubleshooting
- [Arena Analysis](https://causal-judge-evaluation.readthedocs.io/en/latest/guides/arena_analysis.html) - ChatBot Arena-style evaluation

**🔬 Researchers**
- [Theory & Foundations](https://causal-judge-evaluation.readthedocs.io/en/latest/theory/index.html) - Mathematical background
- [API Reference](https://causal-judge-evaluation.readthedocs.io/en/latest/api/index.html) - Complete technical documentation

**🛠️ Developers**
- [Custom Components](https://causal-judge-evaluation.readthedocs.io/en/latest/guides/custom_components.html) - Extend CJE
- [Weight Processing](https://causal-judge-evaluation.readthedocs.io/en/latest/guides/weight_processing.html) - Technical pipeline details

## ⚡ 30-Second Demo

```bash
# Install
pip install cje

# Set your API key
export OPENAI_API_KEY="sk-..."

# Run evaluation
cje run --cfg-path configs --cfg-name arena_test

# View results
cje results --run-dir outputs/arena_test
```

[See complete examples →](https://causal-judge-evaluation.readthedocs.io/en/latest/quickstart.html)

## 🏛️ Architecture

CJE implements a modular pipeline:

```
Data → Log Probabilities → Judge Scores → Causal Estimation → Results
  ↓           ↓                ↓              ↓              ↓
CSV/JSON   Multiple LLMs   Human/AI Judge   DR-CPO/MRDR   Policy Rankings
```

**Key Innovation**: Importance-weighted doubly-robust estimation that corrects for distribution shift between logged and target policies.

## 🔧 Development

```bash
git clone https://github.com/fondutech/causal-judge-evaluation.git
cd causal-judge-evaluation
make dev-setup  # Install in development mode
make docs       # Build documentation locally
make test       # Run test suite
```

## 📊 Performance

Recent benchmarks on ChatBot Arena data:
- **85% MSE reduction** vs naive approaches
- **Catches 100% of teacher forcing bugs** in testing
- **Scales to millions** of logged interactions

## 🤝 Contributing

We welcome contributions! See our [contributing guide](https://causal-judge-evaluation.readthedocs.io/en/latest/contributing.html) and [development documentation](https://causal-judge-evaluation.readthedocs.io/en/latest/api/index.html).

## 📄 License & Citation

MIT License. If you use CJE in research, please [cite our work](https://causal-judge-evaluation.readthedocs.io/en/latest/index.html#citation).

---
**[📖 Read the Full Documentation](https://causal-judge-evaluation.readthedocs.io/)** | **[🚀 Get Started in 5 Minutes](https://causal-judge-evaluation.readthedocs.io/en/latest/quickstart.html)**
