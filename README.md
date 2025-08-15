# Causal Judge Evaluation (CJE) with SIMCal

[![Tests](https://img.shields.io/badge/tests-145%20passing-brightgreen)](cje/tests)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Shape-Constrained, Unbiased Off-Policy Metrics for LLM Systems and Beyond**

CJE transforms routine LLM evaluation logs into unbiased, variance-controlled estimates of counterfactual performance: *"What would our KPI be if we shipped policy π′ instead of π₀?"*

## 🎯 The Problem

Modern LLM evaluation relies on automatic judges (GPT-4, Claude, etc.) to score outputs at scale. But these offline metrics are **correlational**—computed under your logging policy π₀, they don't answer the **causal** question of how a new policy π′ would perform if deployed.

## 💡 The Solution: SIMCal

CJE recasts judge-based evaluation as **calibrated causal inference** using our novel **Surrogate-Indexed Monotone Calibration (SIMCal)**:

### Pipeline Overview

```
[LOGS (π₀)]                                   [ORACLE SLICE (~1-5%)]
• X: context/prompt                           • S: judge score
• A: completion from π₀                       • Y: human label (truth)
• S: judge score                              • Same judge config
• log p_π₀(A|X)                               
     │                                               │
     └──────────────────────────┬────────────────────┘
                                │
                                ▼
                ┌───────────────────────────────────────┐
                │ REWARD CALIBRATION (Isotonic f)       │
                │ • Monotone, mean-preserving (PAV)     │
                │ • Apply R=f(S) to all logs           │
                │ • Targets E[Y(π′)] if (J2-M) holds   │
                └───────────────────────────────────────┘
                                │
                                ▼
                ┌───────────────────────────────────────┐
                │ RAW WEIGHTS W_π′ (TF Cache)           │
                │ • W_π′ = exp(log p_π′ - log p_π₀)    │
                │ • Normalize to mean one (Hájek)       │
                └───────────────────────────────────────┘
                                │
                                ▼
                ┌───────────────────────────────────────┐
                │ SIMCAL: WEIGHT CALIBRATION            │
                │ • Monotone projection indexed by S    │
                │ • OOF stacking to minimize variance   │
                │ • Variance cap ρ: blend → reproject  │
                └───────────────────────────────────────┘
                        │                   │
        ┌───────────────┘                   └────────────────┐
        ▼                                                     ▼
┌─────────────────────┐                      ┌──────────────────────────┐
│ CAL-IPS             │                      │ DR-CPO / TMLE            │
│ V̂ = mean(W_c × R)   │                      │ V̂ = ĝ + W_c×(R−q̂_b)     │
└─────────────────────┘                      └──────────────────────────┘
                        │                   │
                        └─────────┬─────────┘
                                  ▼
                    ┌───────────────────────────┐
                    │ DIAGNOSTICS & GATES       │
                    │ • ESS, Hill index         │
                    │ • Judge reliability       │
                    │ • Orthogonality tests     │
                    └───────────────────────────┘
```

### Key Components

1. **Isotonic Reward Calibration**: Maps judge scores S to calibrated rewards R = f(S) using a small oracle slice
2. **Variance-Safe Weight Calibration**: Projects importance weights onto monotone functions indexed by the judge, with an explicit variance cap
3. **Out-of-Fold Stacking**: Combines {baseline, increasing, decreasing} candidates to minimize influence-function variance
4. **Doubly Robust Estimation**: Achieves √n-rate inference when either nuisance converges at n^(-1/4)

## 📊 Key Results

- **Variance Reduction**: SIMCal increases ESS (effective sample size) by construction through majorization
- **Mean Preservation**: All calibrations preserve the population mean exactly
- **Efficiency**: DR-CPO achieves semiparametric efficiency under standard conditions
- **Auditability**: Comprehensive diagnostics expose assumptions with quantitative gates

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/fondutech/causal-judge-evaluation.git
cd causal-judge-evaluation
pip install -e .
```

### Basic Usage

```python
from cje import analyze_dataset

# One-line causal evaluation
results = analyze_dataset(
    "logs.jsonl",
    estimator="calibrated-ips",  # Uses SIMCal by default
    oracle_coverage=0.1  # 10% oracle labels for calibration
)

# Get policy value estimate with 95% CI
print(f"Policy value: {results.estimates[0]:.3f} ± {1.96 * results.standard_errors[0]:.3f}")
```

### Data Format

CJE expects JSONL logs with:
```json
{
  "prompt": "What is machine learning?",
  "response": "Machine learning is...",
  "base_policy_logprob": -35.704,
  "target_policy_logprobs": {
    "policy_a": -32.456,
    "policy_b": -33.789
  },
  "metadata": {
    "judge_score": 0.85,      // Required: automatic judge score
    "oracle_label": 0.90       // Optional: ground truth (for calibration slice)
  }
}
```

## 🔬 Core Components

### 1. Calibrated IPS (Cal-IPS)
Our primary estimator using SIMCal:

```python
from cje import load_dataset_from_jsonl, calibrate_dataset, PrecomputedSampler, CalibratedIPS

# Load and calibrate
dataset = load_dataset_from_jsonl("logs.jsonl")
calibrated_dataset, cal_result = calibrate_dataset(
    dataset,
    judge_field="judge_score",
    oracle_field="oracle_label",
    oracle_coverage=0.1  # Use 10% for calibration
)

# Run Cal-IPS with SIMCal
sampler = PrecomputedSampler(calibrated_dataset)
estimator = CalibratedIPS(sampler, calibrator=cal_result.calibrator)
results = estimator.fit_and_estimate()
```

### 2. Doubly Robust (DR-CPO)
Sequence-aware doubly robust estimation:

```python
from cje import DRCPOEstimator

# DR-CPO with cross-fitted outcome models
dr_estimator = DRCPOEstimator(
    sampler,
    calibrator=cal_result.calibrator,
    n_folds=5
)

# Add fresh draws (one decode per context)
dr_estimator.add_fresh_draws("policy_a", fresh_draws)
results = dr_estimator.fit_and_estimate()
```

### 3. SIMCal Weight Calibration

The heart of variance control:

```python
from cje.calibration import SIMCalibrator

# Fit SIMCal with variance cap
simcal = SIMCalibrator(
    ordering_index="judge_score",  # Or "calibrated_reward"
    variance_cap=1.0,  # Cap at baseline variance
    candidates=["baseline", "increasing", "decreasing"]
)

# Calibrate weights
calibrated_weights = simcal.fit_transform(
    raw_weights,
    judge_scores,
    residuals=rewards  # For IPS
)
```

## 📈 Estimators

| Estimator | Description | When to Use |
|-----------|-------------|-------------|
| **CalibratedIPS** | IPS with SIMCal weight calibration | Default choice; best variance control |
| **RawIPS** | Standard importance sampling | Baseline comparison |
| **DRCPOEstimator** | Doubly robust with isotonic models | When outcome models available |
| **MRDREstimator** | Policy-specific weighted models | Heterogeneous treatment effects |
| **TMLEEstimator** | Targeted minimum loss | Optimal bias-variance tradeoff |
| **MRDRTMLEEstimator** | MRDR + TMLE targeting | Best of both approaches |

## 🔍 Diagnostics & Gates

CJE provides comprehensive diagnostics to audit assumptions:

### Overlap & Weights
- **ESS (Effective Sample Size)**: Must exceed threshold (default: 1000)
- **Tail Index**: Hill estimator flags heavy tails
- **Overlap Heatmaps**: Visualize support overlap

### Judge Calibration
- **Reliability Diagrams**: Isotonic calibration curves
- **Kendall-τ Drift Test**: Detect judge instability
- **Coverage Checks**: Ensure evaluation within calibration support

### Doubly Robust
- **Orthogonality Score**: Should contain zero
- **EIF Residuals**: Q-Q plots for normality
- **DM-IPS Split**: Component contributions

```python
# Access diagnostics
diagnostics = results.diagnostics
print(diagnostics.summary())

# Check gates
if diagnostics.ess < 1000:
    warnings.warn("Low ESS - consider tighter variance cap")
if diagnostics.tail_index < 2:
    warnings.warn("Heavy tails detected - results may be unstable")
```

## 🧮 Theory & Assumptions

### Core Assumptions

1. **Logging (D1-D3)**: i.i.d. logging under fixed π₀ with overlap
2. **Judge (J1-J2)**: Monotone sufficiency with oracle slice
3. **Regularity (R1-R3)**: Moment conditions and nuisance rates

### Key Theoretical Results

Under assumptions:
- **Mean Preservation**: Isotonic calibration preserves E[Y]
- **Variance Dominance**: SIMCal weakly reduces variance by majorization
- **√n-Normality**: Cal-IPS and DR-CPO achieve asymptotic normality
- **Efficiency**: DR-CPO attains semiparametric efficiency bound

See the paper (forthcoming) for complete theory.

## 🛠️ Advanced Features

### Oracle Coverage Experiments
```python
# Sweep oracle coverage levels
for coverage in [0.05, 0.10, 0.25]:
    results = analyze_dataset(
        "logs.jsonl",
        oracle_coverage=coverage,
        estimator="calibrated-ips"
    )
    print(f"Coverage {coverage}: CI width = {results.ci_width:.3f}")
```

### Variance Cap Sensitivity
```python
# Test different variance caps
estimator = CalibratedIPS(
    sampler,
    variance_cap=0.5  # Tighter cap for more stability
)
```

### Custom Judge Calibration
```python
from cje.calibration import IsotonicCalibrator

# Fit custom calibration
calibrator = IsotonicCalibrator(increasing=True)
calibration_map = calibrator.fit(
    judge_scores[oracle_mask],
    oracle_labels[oracle_mask]
)
calibrated_rewards = calibration_map.transform(judge_scores)
```

## 📚 Documentation

- **Paper**: Forthcoming
- **[API Reference](docs/)**: Full documentation
- **[Diagnostics Guide](cje/diagnostics/)**: Comprehensive diagnostic system
- **[Examples](examples/)**: Jupyter notebooks with tutorials
- **[Arena Experiments](cje/experiments/arena_10k_simplified)**: Production pipeline

## 🧪 Testing

```bash
# Run all tests (145 tests)
poetry run pytest cje/tests/

# Run by category
poetry run pytest -m unit          # Fast unit tests
poetry run pytest -m integration   # Integration tests

# With coverage
poetry run pytest --cov=cje --cov-report=html
```

## 🔄 Beyond LLMs

While designed for LLM evaluation, SIMCal is a general-purpose OPE stabilizer for any problem with:
- Logged propensities
- A one-dimensional surrogate index
- Need for variance control

Applications include:
- Clinical trials with surrogate endpoints
- A/B testing with intermediate metrics
- Recommendation systems with implicit feedback

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

This work builds on foundational research in:
- Off-policy evaluation (Horvitz-Thompson, Dudík et al.)
- Isotonic calibration (van der Laan et al. 2025)
- Semiparametric efficiency (Bickel et al., Chernozhukov et al.)

## 🔗 Links

- **Repository**: This repository
- **Documentation**: See [docs/](docs/) folder
- **Paper**: Forthcoming
- **PyPI**: Coming soon

---

**Made with ❤️ for better offline evaluation**