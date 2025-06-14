# CJE (Causal Judge Evaluation) Implementation Guide

## Overview

CJE (Causal Judge Evaluation) is a Python library for performing off-policy evaluation of Large Language Models (LLMs) using causal inference techniques. It enables unbiased evaluation of new model configurations, prompts, or parameters using only historical interaction data, without requiring deployment.

## Core Concepts

### 1. Off-Policy Evaluation (OPE)
- **Problem**: Evaluate how a new policy π' (target policy) would perform using data collected from a different policy π₀ (logging/behavior policy)
- **Solution**: Use importance sampling and doubly-robust methods to reweight historical data

### 2. Key Components
- **Logging Policy (π₀)**: The model/configuration that generated the historical data
- **Target Policies (π')**: The new models/configurations you want to evaluate
- **Judge**: A model that scores response quality (0-1 scale)
- **Estimators**: Statistical methods that compute unbiased performance estimates

## Architecture

### Directory Structure
```
cje/
├── config/           # Configuration system
│   ├── unified.py    # Main config classes (CJEConfig, PolicyConfig, etc.)
│   └── simple.py     # Simplified config API
├── estimators/       # Statistical estimators
│   ├── ips.py        # IPS and SNIPS estimators
│   ├── drcpo.py      # Doubly-robust CPO estimator
│   ├── mrdr.py       # Model-regularized doubly-robust
│   └── results.py    # EstimationResult class
├── judge/            # Response quality evaluation
│   ├── api_judge.py  # API-based judges (OpenAI, Anthropic, etc.)
│   ├── providers/    # Provider-specific implementations
│   └── factory.py    # Judge creation factory
├── loggers/          # Policy implementations
│   ├── api_policy.py # API-based policy runner
│   ├── multi_target_sampler.py  # Multi-policy importance sampling
│   └── adapters.py   # Provider adapters
├── data/             # Dataset handling
│   ├── base.py       # Base dataset class
│   ├── chatbot_arena.py  # ChatBot Arena dataset
│   └── schema.py     # Data schemas
├── utils/            # Utilities
│   ├── weight_diagnostics.py  # Importance weight analysis
│   ├── inference_cache.py     # LLM response caching
│   └── generation.py          # Response generation utilities
├── cli/              # Command-line interface
│   └── run_experiment.py      # Main experiment runner
└── core.py           # Simplified API entry point
```

### Key Classes and Their Relationships

#### 1. Configuration System (`config/unified.py`)
```python
@dataclass
class CJEConfig:
    paths: PathsConfig
    dataset: DatasetConfig
    logging_policy: PolicyConfig      # Required: what generated the data
    target_policies: List[TargetPolicyConfig]  # Required: what to evaluate
    judge: JudgeConfig               # Required: how to score responses
    estimator: EstimatorConfig       # Required: statistical method
    oracle: Optional[OracleConfig]   # Optional: ground truth labels
```

The configuration system uses a builder pattern (`ConfigurationBuilder`) for programmatic construction and supports YAML loading via `from_dict()`.

#### 2. Data Flow Pipeline

```
1. Load Dataset → CJEDataset object with contexts
2. Generate/Load Responses → Add responses from logging policy
3. Compute Log Probabilities → Add log p(response|context) under π₀
4. Score Responses → Judge assigns quality scores (0-1)
5. Compute Importance Weights → w = π'(response|context) / π₀(response|context)
6. Run Estimation → Apply statistical estimator to get final results
```

#### 3. Estimator Hierarchy (`estimators/`)

All estimators inherit from `Estimator[T]` base class:
- `MultiIPSEstimator`: Basic importance sampling
- `MultiSNIPSEstimator`: Self-normalized importance sampling
- `MultiDRCPOEstimator`: Doubly-robust with cross-fitted outcome models
- `MultiMRDREstimator`: Advanced doubly-robust with variance optimization

Each returns an `EstimationResult` with:
- `v_hat`: Point estimates for each target policy
- `se`: Standard errors
- `covariance_matrix`: Full covariance matrix
- `confidence_interval()`: Method to compute CIs

#### 4. Provider System (`providers/` and `judge/providers/`)

Unified provider interface supporting:
- **API Providers**: OpenAI, Anthropic, Google, Fireworks, Together
- **Local Models**: HuggingFace (`hf` provider)
- **Mock Provider**: For testing

Each provider implements:
- Response generation with log probabilities
- Judge scoring
- Automatic retries and error handling

#### 5. Multi-Target Sampling (`loggers/multi_target_sampler.py`)

Critical component that handles:
- Computing importance weights for K target policies simultaneously
- Numerical stabilization (log-space computation)
- Weight clipping and diagnostics
- Effective Sample Size (ESS) computation

### Important Implementation Details

#### 1. Weight Computation and Numerical Stability

```python
# In MultiTargetSampler.importance_weights_matrix()
# Compute in log space for numerical stability
log_ratios = target_logps - behavior_logps  # Shape: (n, K)

# Apply clipping in log space BEFORE exponentiation
if clip is not None:
    log_ratios = np.clip(log_ratios, -clip, clip)

# Stabilize extreme values
if stabilize:
    log_ratios = self._stabilize_log_ratios(log_ratios)

# Convert to weights
weights = np.exp(log_ratios)
```

#### 2. Cross-Fitting in DR Estimators

DR estimators use k-fold cross-fitting to avoid overfitting:
1. Split data into k folds
2. For each fold:
   - Train outcome model on other k-1 folds
   - Predict on held-out fold
3. Combine predictions for final estimate

#### 3. Judge Templates (`prompts/unified_templates.py`)

Pre-defined judge templates:
- `quick_judge`: Fast 0-10 scoring
- `comprehensive_judge`: Detailed evaluation
- `helpfulness_0_10`: Specific criteria
- Custom templates supported via `custom_template` field

#### 4. Caching System (`utils/inference_cache.py`)

SQLite-based cache for:
- LLM generations (responses + log probs)
- Judge scores
- Keyed by (model, prompt, response) tuple
- Automatic cache directory: `.cache/llm_cache.sqlite`

### Configuration Examples

#### Minimal Configuration
```yaml
dataset:
  name: "./data.csv"

logging_policy:
  provider: "openai"
  model_name: "gpt-3.5-turbo"

target_policies:
  - name: "improved"
    provider: "openai"
    model_name: "gpt-4-turbo"

judge:
  provider: "openai"
  model_name: "gpt-4-turbo"
  template: "quick_judge"

estimator:
  name: "DRCPO"
  k: 5
```

#### Advanced Features

1. **Oracle Mode**: Use ground truth labels for calibration
```yaml
oracle:
  enabled: true
  provider: "openai"
  model_name: "gpt-4-turbo"
  logging_policy_oracle_fraction: 0.25  # 25% for calibration
```

2. **Weight Diagnostics**: Monitor importance weight health
```yaml
diagnostics:
  log_ratio_clip: 20.0  # ±20 in log space
  ess_warning_threshold: 15.0  # Warn if ESS < 15% of n
  save_diagnostic_plots: true
```

3. **Multiple Target Policies**: Evaluate many policies at once
```yaml
target_policies:
  - name: "gpt4"
    provider: "openai"
    model_name: "gpt-4-turbo"
  - name: "claude"
    provider: "anthropic"
    model_name: "claude-3-sonnet-20240229"
  - name: "local"
    provider: "hf"
    model_name: "meta-llama/Llama-2-7b-chat-hf"
```

### Entry Points

#### 1. CLI (`cje` command)
```bash
cje run --cfg-path configs --cfg-name experiment
```

#### 2. Pipeline API
```python
from cje.pipeline import run_pipeline
results = run_pipeline(cfg_path="configs", cfg_name="experiment")
```

#### 3. Simplified Core API
```python
from cje.core import run_cje
result = run_cje("config.yaml")
print(result.summary())
```

#### 4. Programmatic Configuration
```python
from cje.config import simple_config

config = simple_config(
    logging_model="gpt-3.5-turbo",
    logging_provider="openai",
    target_models=["gpt-4-turbo"],
    target_providers=["openai"],
    target_names=["gpt4"],
    judge_model="gpt-4-turbo",
    judge_provider="openai"
)
results = config.run()
```

### Key Algorithms

#### 1. IPS (Inverse Propensity Scoring)
```
V̂(π') = (1/n) Σᵢ wᵢ · rᵢ
where wᵢ = π'(aᵢ|xᵢ) / π₀(aᵢ|xᵢ)
```

#### 2. SNIPS (Self-Normalized IPS)
```
V̂(π') = Σᵢ wᵢ · rᵢ / Σᵢ wᵢ
```

#### 3. DR-CPO (Doubly Robust)
```
V̂(π') = (1/n) Σᵢ [μ̂(xᵢ) + wᵢ(rᵢ - μ̂(xᵢ,aᵢ))]
```
With cross-fitting and optional isotonic calibration.

#### 4. MRDR (Model-Regularized DR)
Minimizes asymptotic variance by optimizing the outcome model specifically for importance-weighted estimation.

### Critical Design Decisions

1. **All Estimators are Multi-Policy**: Single-policy evaluation is just K=1 case
2. **Log-Space Computation**: All probability ratios computed in log space
3. **Unified Provider Interface**: Same interface for all LLM providers
4. **Mandatory Fields**: No optional inference - explicit configuration required
5. **Builder Pattern**: Clean programmatic configuration API
6. **Result Standardization**: All estimators return `EstimationResult`

### Common Pitfalls and Solutions

1. **Distribution Shift**: Use DR methods (DRCPO/MRDR) for robustness
2. **Low ESS**: Indicates high variance - check weight diagnostics
3. **Numerical Overflow**: Always use log-space computation
4. **Cache Invalidation**: Delete `.cache/` directory if needed
5. **Provider Errors**: Built-in retry logic with exponential backoff

### Environment Variables

```bash
# API Keys
OPENAI_API_KEY="sk-..."
ANTHROPIC_API_KEY="sk-ant-..."
GOOGLE_API_KEY="..."
FIREWORKS_API_KEY="..."
TOGETHER_API_KEY="..."

# Cache Control
CJE_CACHE_DIR="./my_cache"
CJE_OUTPUT_DIR="./outputs"
```

### Testing and Validation

The library includes:
- Unit tests for all components
- Integration tests for full pipeline
- Mock providers for testing without API calls
- Validation commands: `cje validate config` and `cje validate data`

### Performance Considerations

1. **Caching**: All LLM calls cached by default
2. **Parallel Processing**: Cross-fitting uses joblib parallelization
3. **Batch Processing**: HuggingFace provider supports batching
4. **Memory Efficiency**: Streaming data processing for large datasets

This implementation guide should provide a comprehensive understanding of the CJE library's architecture, design decisions, and usage patterns.