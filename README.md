# Causal Judge Evaluation (CJE) with SIMCal

[![Tests](https://img.shields.io/badge/tests-145%20passing-brightgreen)](cje/tests)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Shape-Constrained, Unbiased Off-Policy Metrics for LLM Systems and Beyond**

CJE transforms routine LLM evaluation logs into unbiased, variance-controlled estimates of counterfactual performance: *"What would our KPI be if we shipped policy π′ instead of π₀?"*

## 🎯 The Problem

Modern LLM evaluation relies on automatic judges (GPT-4, Claude, etc.) to score outputs at scale. But these offline metrics are **correlational**—computed under your logging policy π₀, they don't answer the **causal** question of how a new policy π′ would perform if deployed.

## 💡 The Solution: SIMCal

CJE recasts judge-based evaluation as **calibrated causal inference** using our novel **Surrogate-Indexed Monotone Calibration (SIMCal)** - providing unbiased estimates with explicit variance control.

## 🚀 Quick Start

```bash
# Install
pip install causal-judge-evaluation  # Coming soon to PyPI
# Or: pip install -e git+https://github.com/fondutech/causal-judge-evaluation.git

# Set API key for log probability computation
export FIREWORKS_API_KEY="your-api-key"
```

```python
from cje import analyze_dataset

# One-line causal evaluation
results = analyze_dataset(
    "logs.jsonl",
    estimator="calibrated-ips",  # Uses SIMCal by default
    oracle_coverage=0.1  # 10% oracle labels for calibration
)

print(f"Policy value: {results.estimates[0]:.3f} ± {1.96 * results.standard_errors[0]:.3f}")
```

## 📊 How It Works

CJE transforms biased judge scores into unbiased policy estimates through a principled pipeline:

```
┌──────────────────────────────────────────────────────────────────────┐
│                            INPUT DATA                                │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Logged Conversations                      │    │
│  │  • Prompts (X)                                              │    │
│  │  • Responses (A) from policy π₀                            │    │
│  │  • Judge scores (S) from automatic evaluator                │    │
│  │  • Log probabilities: log p_π₀(A|X)                         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                │                                     │
│          ┌─────────────────────┴──────────────────────┐             │
│          ▼                                            ▼             │
│  ┌───────────────┐                          ┌────────────────────┐  │
│  │ Oracle Subset │                          │ Full Dataset       │  │
│  │ (~5-10% data) │                          │ (100% of data)     │  │
│  │ + Human labels│                          │ Judge scores only  │  │
│  │     (Y)       │                          │                    │  │
│  └───────────────┘                          └────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
           │                                            │
           ▼                                            │
  ┌──────────────────────────┐                         │
  │   CALIBRATION STEP       │                         │
  │                          │                         │
  │ Learn f: S → Y via       │                         │
  │ isotonic regression      │                         │
  │ • Cross-fit with k folds │                         │
  │ • f^(-k) for fold k      │                         │
  └──────────────────────────┘                         │
           │                                            │
           └──────────────┬─────────────────────────────┘
                          ▼
                ┌─────────────────────┐        ┌──────────────────────┐
                │ Apply Calibration   │        │ Compute Importance   │
                │                     │        │      Weights         │
                │ R_i = f^(-k_i)(S_i)│        │                      │
                │ for ALL samples    │  ────►  │ W_i = p_π′(A_i|X_i) │
                │ using cross-fitted │        │       ────────────   │
                │ models             │        │       p_π₀(A_i|X_i)  │
                └─────────────────────┘        └──────────────────────┘
                          │                              │
                          ▼                              ▼
                ┌─────────────────────┐        ┌──────────────────────┐
                │ Calibrated Rewards │        │  SIMCal Projection   │
                │                     │        │                      │
                │ • Unbiased under   │        │ W_c = Proj_𝒮(W)     │
                │   monotonicity     │        │                      │
                │ • All samples have │        │ • 𝒮 = monotone fns   │
                │   calibrated R     │        │   indexed by S       │
                └─────────────────────┘        │ • Mean preserving    │
                          │                    │ • Var(W_c) ≤ Var(W)  │
                          │                    └──────────────────────┘
                          │                              │
                          └──────────┬───────────────────┘
                                     ▼
                          ┌──────────────────────┐
                          │   FINAL ESTIMATE    │
                          │                      │
                          │  IPS Estimator:      │
                          │  V̂ = Σ W_c,i × R_i/n │
                          │                      │
                          │  DR (if fresh draws):│
                          │  V̂ = ĝ + Σ W_c,i ×  │
                          │      (R_i - q̂_i)/n  │
                          │                      │
                          │  SE = std(ψ̂)/√n     │
                          │  95% CI: V̂ ± 1.96×SE│
                          └──────────────────────┘

Legend: X=prompts, A=responses, S=judge scores, Y=oracle labels, R=rewards
        W=importance weights, W_c=calibrated weights, ψ̂=influence function
```

### Key Innovation: SIMCal

SIMCal prevents weight explosion by projecting importance weights onto monotone functions indexed by the judge score, ensuring:
- ✅ Mean preservation (unbiasedness)
- ✅ Variance reduction via majorization
- ✅ Explicit variance cap for stability

## 💻 Example: Comparing Model Versions

```python
from cje import analyze_dataset

# Evaluate if switching from Llama-3-8B to Llama-3-70B improves quality
results = analyze_dataset(
    "llama_logs.jsonl",  # Your logged Llama-3-8B conversations
    estimator="calibrated-ips",
    oracle_coverage=0.05,  # 5% human labels
    target_policies=["llama-3-70b", "llama-3.1-70b"]  # Compare larger models
)

# Compare policies
for i, policy in enumerate(["llama-3-70b", "llama-3.1-70b"]):
    estimate = results.estimates[i]
    stderr = results.standard_errors[i]
    print(f"{policy}: {estimate:.3f} ± {1.96*stderr:.3f}")
    
# Check diagnostics
if results.diagnostics.ess < 1000:
    print("⚠️ Warning: Low effective sample size - consider more data")
```

## 📋 Data Format

CJE expects JSONL logs with:
```json
{
  "prompt": "What is machine learning?",
  "response": "Machine learning is...",
  "base_policy_logprob": -35.704,  // Log P(response|prompt) under π₀
  "target_policy_logprobs": {       // Optional: pre-computed for efficiency
    "policy_a": -32.456,
    "policy_b": -33.789
  },
  "metadata": {
    "judge_score": 0.85,      // Required: automatic judge score
    "oracle_label": 0.90       // Optional: ground truth (for calibration slice)
  }
}
```

**Note**: For DR estimators, you'll also need fresh draws (new responses from π′ evaluated by the judge). See [examples](examples/) for details.

## 🔬 Available Estimators

| Estimator | Description | When to Use |
|-----------|-------------|-------------|
| **CalibratedIPS** | IPS with SIMCal weight calibration | Logged data only; best IPS variant |
| **RawIPS** | Standard importance sampling | Baseline comparison; diagnostic purposes |
| **DRCPOEstimator** | Doubly-Robust Counterfactual Policy Optimization | When you have fresh draws; generally best |
| **MRDREstimator** | More Robust Doubly-Robust estimator | Fresh draws + concern about misspecification |
| **TMLEEstimator** | Targeted maximum likelihood | Fresh draws + want optimal efficiency |
| **MRDRTMLEEstimator** | MRDR + TMLE targeting | Best of both: robustness + efficiency |

## 🔍 Diagnostics & Quality Gates

CJE provides comprehensive diagnostics to audit assumptions:

```python
diagnostics = results.diagnostics
print(diagnostics.summary())

# Automatic quality gates
if diagnostics.ess < 1000:
    warnings.warn("Low ESS - consider tighter variance cap")
if diagnostics.tail_index < 2:
    warnings.warn("Heavy tails detected - results may be unstable")
```

Key diagnostics include:
- **ESS (Effective Sample Size)**: Overlap quality metric
- **Tail Index**: Heavy tail detection via Hill estimator
- **Calibration Curves**: Judge reliability visualization
- **Orthogonality Scores**: DR assumption checking

## 🛠️ Advanced Usage

### Multiple Estimators
```python
# Run multiple estimators for robustness
for estimator in ["calibrated-ips", "dr-cpo", "tmle"]:
    results = analyze_dataset("logs.jsonl", estimator=estimator)
    print(f"{estimator}: {results.estimates[0]:.3f}")
```

### Custom Calibration
```python
from cje import load_dataset_from_jsonl, calibrate_dataset, PrecomputedSampler, CalibratedIPS

# Fine-grained control
dataset = load_dataset_from_jsonl("logs.jsonl")
calibrated_dataset, cal_result = calibrate_dataset(
    dataset,
    judge_field="judge_score",
    oracle_field="oracle_label",
    oracle_coverage=0.1
)

sampler = PrecomputedSampler(calibrated_dataset)
estimator = CalibratedIPS(sampler, variance_cap=0.5)  # Tighter cap
results = estimator.fit_and_estimate()
```

## 📚 Documentation

- **[Theory & Methods](docs/THEORY.md)**: Mathematical foundations and detailed pipeline
- **[Examples](examples/)**: Jupyter notebooks with tutorials

### Module Documentation
- **[Calibration](cje/calibration/)**: SIMCal algorithm and isotonic regression
- **[Data](cje/data/)**: Data models and validation
- **[Teacher Forcing](cje/teacher_forcing/)**: Log probability computation
- **[Estimators](cje/estimators/)**: All estimation methods

## 🧪 Testing

```bash
# Run all tests
poetry run pytest cje/tests/

# With coverage
poetry run pytest --cov=cje --cov-report=html
```

## 🔑 Key Concepts

- **π₀**: Your logging/base policy (what generated the data)
- **π′**: Target policy you want to evaluate
- **Judge Score (S)**: Automatic evaluation score (e.g., from GPT-4)
- **Oracle Label (Y)**: Human ground truth for calibration
- **SIMCal**: Our variance reduction method via monotone projection
- **ESS**: Effective Sample Size - measures overlap quality

## 📖 Citation

If you use CJE in your research, please cite:
```bibtex
@software{cje2025,
  title={Causal Judge Evaluation (CJE): A Framework for Unbiased Off-Policy 
         Evaluation with Shape-Constrained Calibration},
  author={CJE Contributors},
  year={2025},
  url={https://github.com/causal-judge-evaluation}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

This work builds on foundational research in off-policy evaluation (Horvitz-Thompson, Dudík et al.), isotonic calibration (van der Laan et al. 2025), and semiparametric efficiency (Bickel et al., Chernozhukov et al.).

---

**Made with ❤️ for better offline evaluation**