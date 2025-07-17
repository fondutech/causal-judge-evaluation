# CLAUDE.md

Core guidance for Claude Code when working with the CJE repository.

Last updated: 2025-01-17 (SOLID refactoring completed)

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
├── core/                 # Core abstractions (estimators, base classes)
├── data/                 # Data models and loading utilities  
├── teacher_forcing/      # Log probability computation
├── utils/                # Calibration, diagnostics, helpers
└── tests/                # Test suite with example data
```

## 🚀 Quick Start

```python
from cje_simplified import load_dataset_with_calibration, PrecomputedSampler, CalibratedIPS

# Modern SOLID approach - use convenience functions
dataset, stats = load_dataset_with_calibration("data.jsonl") 
sampler = PrecomputedSampler(dataset)

# Alternative - direct factory usage with dependency injection
from cje_simplified import DatasetFactory, DatasetLoader
factory = DatasetFactory(loader=DatasetLoader())
dataset = factory.create_from_jsonl("data.jsonl")
sampler = PrecomputedSampler(dataset)

# Run estimation
estimator = CalibratedIPS(sampler)
results = estimator.fit_and_estimate()
```

## 🏗️ Architecture Principles

### 1. Data Models First (Pydantic)
All data structures are defined in `data/models.py`:
- `Sample` - Single data point with validation
- `Dataset` - Pure data container (no loading logic)
- `LogProbResult` - Explicit error handling for API calls
- `EstimationResult` - Structured results with statistical methods

### 2. Separation of Concerns (SOLID)
Loading responsibilities are cleanly separated:
- `DatasetLoader` - Converts raw data to typed Dataset objects
- `DatasetFactory` - Orchestrates loading + calibration workflows
- `DataSource` - Protocol for different data sources (JSONL, memory, etc.)
- `Dataset` - Pure data container with validation only

### 3. Dependency Injection
```python
# Good - explicit dependencies with injection
from cje_simplified import DatasetFactory, DatasetLoader
factory = DatasetFactory(loader=DatasetLoader(base_policy_field="custom_field"))
dataset = factory.create_from_jsonl("data.jsonl")

# Better - use convenience functions for common cases
from cje_simplified import load_dataset_from_jsonl
dataset = load_dataset_from_jsonl("data.jsonl")
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
- `DatasetFactory` - Coordinates loading + calibration
- `PrecomputedSampler` - Adds CJE-specific operations to Dataset
- `BaseCJEEstimator` - Abstract interface for all estimators
- `CalibratedIPS` - Concrete implementation with cross-fitting

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

# Run experiments
cd cje_simplified
python example_usage.py
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
- Rewards must be calibrated to business KPIs (use `load_dataset_with_calibration()` or `DatasetFactory`)

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

# Or with calibration
from cje_simplified import load_dataset_with_calibration
dataset, stats = load_dataset_with_calibration("file.jsonl")
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
from cje_simplified import load_dataset_with_calibration, PrecomputedSampler
dataset, stats = load_dataset_with_calibration("data.jsonl")
sampler = PrecomputedSampler(dataset)

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
- `create_calibrated_rewards()` → `load_dataset_with_calibration()`

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