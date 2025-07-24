# CLAUDE.md

Core guidance for Claude Code when working with the CJE repository.

Last updated: 2025-01-20 (Unified evaluation system, simplified data pipeline)

## 🎯 Project Philosophy

The original CJE codebase became a "ball of mud" - overly complex, tightly coupled, and difficult to maintain. We created `cje_simplified/` to start fresh with:
- **Clear separation of concerns** 
- **Type safety with Pydantic models**
- **Dependency injection over hidden state**
- **Simple, composable abstractions**
- **Explicit error handling (no magic fallbacks)**

## 📁 Repository Structure

```
cje/                      # Original codebase (deprecated, do not modify)
cje_simplified/           # Clean reimplementation - ALL NEW WORK GOES HERE
├── calibration/          # Calibration utilities (isotonic, judge, dataset)
├── core/                 # Core abstractions (estimators, base classes)
├── data/                 # Data models and loading utilities  
├── teacher_forcing/      # Log probability computation
├── utils/                # Diagnostics and helpers
└── tests/                # Test suite with example data
```

## 🚀 Quick Start

```python
from cje_simplified import load_dataset_from_jsonl, calibrate_dataset, PrecomputedSampler, CalibratedIPS

# Three distinct workflows:

# 1. Oracle labels as rewards (no calibration needed)
dataset = load_dataset_from_jsonl("data_with_oracle.jsonl")  # Assumes oracle_label field exists
# Map oracle labels directly to rewards
for sample in dataset.samples:
    sample.reward = sample.metadata["oracle_label"]

# 2. Judge scores that need calibration
dataset = load_dataset_from_jsonl("data_with_judges.jsonl")  # No rewards yet
calibrated_dataset, result = calibrate_dataset(
    dataset, 
    judge_field="judge_score",
    oracle_field="oracle_label"
)

# 3. Pre-calibrated rewards
dataset = load_dataset_from_jsonl("data_with_rewards.jsonl")  # Already has reward field

# Run estimation (requires rewards)
sampler = PrecomputedSampler(dataset)  # Will error if rewards are missing
estimator = CalibratedIPS(sampler)
results = estimator.fit_and_estimate()
```

## 🏗️ Architecture Principles

### 1. Data Models First (Pydantic)
All data structures are defined in `data/models.py`:
- `Sample` - Single data point with validation (reward is Optional)
- `Dataset` - Pure data container (no loading logic)
- `LogProbResult` - Explicit error handling for API calls
- `EstimationResult` - Structured results with statistical methods

**Key change**: Rewards are now optional in Sample, allowing datasets to exist before calibration.

### 2. Separation of Concerns (SOLID)
Responsibilities are cleanly separated:
- `DatasetLoader` - Converts raw data to typed Dataset objects
- `DatasetFactory` - Creates datasets from various sources
- `calibrate_dataset()` - Calibrates judge scores to oracle labels
- `DataSource` - Protocol for different data sources (JSONL, memory, etc.)
- `Dataset` - Pure data container with validation only

### 3. Modular Data Pipeline
```python
# Load data without rewards (new default behavior)
dataset = load_dataset_from_jsonl("data.jsonl")  # No rewards required

# Calibrate when needed (separate step)
if has_judge_scores:
    dataset, stats = calibrate_dataset(dataset, judge_field="judge_score", oracle_field="oracle_label")

# Or directly assign oracle labels as rewards
if has_oracle_labels:
    for sample in dataset.samples:
        sample.reward = sample.metadata["oracle_label"]

# Custom field names with dependency injection
factory = DatasetFactory(loader=DatasetLoader(base_policy_field="p0_logprob"))
dataset = factory.create_from_jsonl("data.jsonl")

# PrecomputedSampler validates rewards exist
sampler = PrecomputedSampler(dataset)  # Errors if rewards are None
```

### 4. No Magic Fallbacks
```python
# Good - explicit None for failures
if sample.base_policy_logprob is None:
    return None

# Bad - magic fallback values
return -100.0  # NEVER DO THIS
```

### 5. Clear Abstractions (Single Responsibility)
- `Dataset` - Data container with validation only
- `DatasetLoader` - Converts raw data to Dataset objects  
- `DatasetFactory` - Creates datasets from various sources
- `calibration/` - All calibration functionality
  - `isotonic.py` - Cross-fitted isotonic regression utilities
  - `judge.py` - Judge score calibration to oracle labels
  - `dataset.py` - Dataset calibration workflows
- `PrecomputedSampler` - Adds CJE-specific operations to Dataset
- `BaseCJEEstimator` - Abstract interface for all estimators
- `CalibratedIPS` - Concrete implementation with cross-fitting

### 6. Simplified Chat API for Teacher Forcing
```python
from cje_simplified import compute_chat_logprob, Llama3TemplateConfig

# Explicit template configuration (no auto-detection)
config = Llama3TemplateConfig()  # or HuggingFaceTemplateConfig("model-name")

# Compute log probability for chat
result = compute_chat_logprob(
    chat=[
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"}
    ],
    model="accounts/fireworks/models/llama-v3-8b-instruct",
    template_config=config  # Explicit, no magic
)
```

## 🔧 Essential Commands

```bash
# Development setup
make dev-setup

# Run tests
poetry run pytest                     # Fast tests only
poetry run pytest --run-slow          # Include slow tests
poetry run pytest cje_simplified/     # Test simplified codebase only

# Linting (MUST pass before commits)
make lint

# Run experiments (typical workflow)
cd cje_simplified/experiments/arena_10k_simplified
python generate_responses.py --prompts data/prompts.jsonl
python add_judge_scores.py --input data/responses/base_responses.jsonl
python add_oracle_labels.py --input data/responses/base_responses.jsonl
python compute_logprobs.py --responses-dir data/responses
python prepare_cje_data.py --responses-dir data/responses --logprobs-dir data/logprobs
python run_cje_analysis.py --data data/cje_dataset.jsonl
```

## 🔑 API Keys

Always source secrets before running:
```bash
source /Users/eddielandesberg/PycharmProjects/causal-judge-evaluation/set_secrets.sh
```

## 📊 Data Format

Expected JSONL format:
```json
{
  "prompt": "What is 2+2?",
  "response": "4",
  "reward": 0.85,              // Calibrated reward in [0,1]
  "base_policy_logprob": -35.704,  // Base policy log P(response)
  "target_logps": {
    "pi_improved": -32.123,
    "pi_baseline": -36.456
  }
}
```

**Critical**: 
- Use `base_policy_logprob` for the base policy field
- Store failed log probs as `null`, never use fallback values
- Rewards must be calibrated to business KPIs (use `calibrate_dataset()` after loading)

## 📝 Key Architectural Changes (Latest)

1. **Decoupled Loading and Calibration**
   - Removed `load_dataset_with_calibration()` 
   - Loading and calibration are now separate operations
   - Enables three distinct workflows (oracle, judge calibration, pre-calibrated)

2. **Optional Rewards**
   - `Sample.reward` is now `Optional[float]`
   - Datasets can be loaded without rewards
   - `PrecomputedSampler` validates rewards exist before estimation

3. **Simplified Chat API**
   - Removed `ChatTeacherForcing` class
   - Removed auto-detection in favor of explicit template configuration
   - Reduced from 6 layers to 2-3 layers

4. **Calibration Module**
   - Created dedicated `calibration/` directory
   - Removed redundant `fit_isotonic_with_cv` function
   - Clear separation of isotonic, judge, and dataset calibration

5. **Metadata Auto-Collection**
   - DatasetLoader automatically puts non-core fields into metadata
   - Enables fields like `judge_score` and `oracle_label` to be accessed uniformly

6. **Unified Evaluation System**
   - Single `FireworksEvaluator` class for both judges and oracles
   - Judges and oracles differ only in model choice
   - Judge model: `accounts/fireworks/models/llama4-scout-instruct-basic`
   - Oracle model: `accounts/fireworks/models/kimi-k2-instruct`
   - Uses LangChain structured outputs for reliable scoring (0-100 scale)
   - Minimal scripts: `add_judge_scores.py` and `add_oracle_labels.py`
   - Oracle labels all responses for validation purposes

7. **Experiment Pipeline Data Flow**
   - Response files are modified in place to add evaluation scores
   - `prepare_cje_data.py` reads from both response and logprob files
   - Judge/oracle scores stored in `metadata` field of response files
   - Log probabilities computed for BASE responses under all policies
   - Uses median of 3 logprob samples to handle API non-determinism
   - Final dataset combines everything for CJE analysis

8. **Pre-computed Rewards in Arena Experiment**
   - `prepare_cje_data.py` supports `--oracle-coverage` parameter (0.0 to 1.0)
   - When 1.0: Uses oracle labels directly as rewards
   - When < 1.0: Calibrates judge scores using random subset of oracle labels
   - `run_cje_analysis.py` checks for pre-computed rewards first
   - Base policy results included in output (marked as "observed" vs "counterfactual")
   - Enables ablation studies on oracle coverage without separate scripts

## ⚠️ Common Pitfalls

### 1. Wrong Log Prob Field
```python
# Wrong - old field names
"total_logprob": -35.704
"p0_logprob": -35.704

# Correct - standard field name
"base_policy_logprob": -35.704
```

### 2. Magic Fallback Values
```python
# Wrong - corrupts analysis
logprob = api_result or -100.0

# Correct - explicit failure
logprob = api_result  # None if failed
```

### 3. Using Old Loading Patterns
```python
# Wrong - old methods removed in SOLID refactoring
dataset = Dataset.from_raw_data(data)
dataset = Dataset.from_jsonl("file.jsonl")

# Correct - use factory or convenience functions
from cje_simplified import load_dataset_from_jsonl
dataset = load_dataset_from_jsonl("file.jsonl")

# For calibration - now a separate step
dataset = load_dataset_from_jsonl("file.jsonl")  # No reward_field needed
calibrated_dataset, stats = calibrate_dataset(dataset, judge_field="judge_score")
```

## 🧪 Testing Philosophy

- Small, focused unit tests
- Real data in `tests/data/` for integration tests
- No mocking of core abstractions (use real Dataset objects)
- Test behavior, not implementation details

## 🚨 Red Flags in Code Review

1. **Imports from `cje/`** - Should only import from `cje_simplified/`
2. **Magic numbers** - Like -100.0, 0.0 as fallbacks
3. **try/except with fallbacks** - Prefer explicit None returns
4. **Direct dict manipulation** - Use Pydantic models
5. **Tight coupling** - Each class should have one clear responsibility
6. **Old loading patterns** - Using removed Dataset methods like `.from_raw_data()`
7. **Violating SRP** - Classes with multiple responsibilities (loading + validation + business logic)
8. **Hidden dependencies** - Construction logic inside data models
9. **Coupled loading/calibration** - Using removed `load_dataset_with_calibration()`
10. **Assuming rewards exist** - Not checking if rewards are None before using PrecomputedSampler

## 📈 Importance Weight Monitoring

Watch for extreme weights that indicate problems:
```python
# Check effective sample size
ess = (sum(weights))**2 / sum(w**2 for w in weights)
if ess < n_samples * 0.1:  # Less than 10% effective
    log.error("Effective sample size too low!")

# Monitor extremes
if weight > 100 or weight < 0.01:
    log.warning(f"Extreme weight: {weight}")
```

## 🔄 Migration Guide

Moving from old to new codebase:

```python
# Old approach
from cje import PrecomputedLogger
logger = PrecomputedLogger(data, p0_policy_name="p0")

# New SOLID approach - recommended
from cje_simplified import load_dataset_from_jsonl, calibrate_dataset, PrecomputedSampler
dataset = load_dataset_from_jsonl("data.jsonl")
calibrated_dataset, stats = calibrate_dataset(dataset, judge_field="judge_score")
sampler = PrecomputedSampler(calibrated_dataset)

# New SOLID approach - with dependency injection  
from cje_simplified import DatasetFactory, DatasetLoader
factory = DatasetFactory(loader=DatasetLoader())
dataset = factory.create_from_data(data)
sampler = PrecomputedSampler(dataset)
```

**Field Name Migration:**
- `total_logprob` → `base_policy_logprob`
- `p0_logprob` → `base_policy_logprob`  
- `target_logps` → `target_policy_logprobs`

**Loading Pattern Migration:**
- `Dataset.from_raw_data()` → `DatasetFactory.create_from_data()`
- `Dataset.from_jsonl()` → `load_dataset_from_jsonl()` or `DatasetFactory.create_from_jsonl()`
- `load_dataset_with_calibration()` → `load_dataset_from_jsonl()` + `calibrate_dataset()` (now separate steps)

## 📝 Documentation Standards

- Docstrings for all public methods
- Type hints for all parameters and returns
- Examples in docstrings for complex functionality
- Update this file when adding major features

## 🎓 Key Lessons from the Rewrite

1. **Start with data models** - Define your types first
2. **Apply SOLID principles** - Single responsibility, dependency injection, open/closed
3. **Separate concerns early** - Loading ≠ Validation ≠ Business Logic  
4. **Make dependencies explicit** - No hidden object construction
5. **Fail loudly** - Better to error than silently corrupt results
6. **Keep it simple** - Resist adding "clever" features
7. **Use protocols for extensibility** - Easy to add new data sources
8. **Factory pattern for coordination** - When you need to orchestrate multiple responsibilities

Remember: The goal of `cje_simplified` is to be **simple, correct, and maintainable**. When in doubt, choose clarity over cleverness.