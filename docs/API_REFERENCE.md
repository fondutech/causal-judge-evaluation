# CJE API Reference

## Core Configuration API

### `simple_config()`
Create a simple configuration with sensible defaults.

```python
from cje.config.unified import simple_config

config = simple_config(
    work_dir="./outputs/experiment",
    dataset_name="./data/test.jsonl",
    logging_model="gpt-3.5-turbo",
    logging_provider="openai",
    target_model="gpt-4",
    target_provider="openai",
    judge_model="gpt-4o",
    judge_provider="openai",
    estimator_name="DRCPO"
)
```

### `multi_policy_config()`
Create a configuration for evaluating multiple target policies.

```python
from cje.config.unified import multi_policy_config

config = multi_policy_config(
    dataset_name="./data/test.jsonl",
    target_policies=[
        {"name": "conservative", "temperature": 0.1},
        {"name": "balanced", "temperature": 0.7},
        {"name": "creative", "temperature": 1.2},
    ]
)
```

### `ConfigurationBuilder`
Fine-grained control over configuration.

```python
from cje.config.unified import ConfigurationBuilder

config = (ConfigurationBuilder()
    .paths("./outputs")
    .dataset("ChatbotArena", split="test")
    .logging_policy("gpt-3.5-turbo", provider="openai")
    .add_target_policy("target1", "gpt-4", provider="openai", temperature=0.1)
    .judge("openai", "gpt-4o")
    .estimator("DRCPO")
    .build())
```

## Pipeline API

### Running Experiments

```python
# Using configuration
results = config.run()

# Or using the pipeline directly
from cje.pipeline import CJEPipeline, PipelineConfig

pipeline_config = PipelineConfig(
    work_dir=Path("./outputs"),
    dataset_config={"name": "test.jsonl"},
    logging_policy_config={"provider": "openai", "model_name": "gpt-3.5-turbo"},
    judge_config={"provider": "openai", "model_name": "gpt-4o"},
    target_policies_config=[{"provider": "openai", "model_name": "gpt-4"}],
    estimator_configs=[{"name": "DRCPO", "params": {}}]
)

pipeline = CJEPipeline(pipeline_config)
results = pipeline.run()
```

## Estimators

### Available Estimators

- **IPS**: Inverse Propensity Scoring
- **SNIPS**: Self-Normalized IPS  
- **CalibratedIPS**: IPS with propensity calibration
- **DRCPO**: Doubly-Robust Cross-Policy Optimization
- **MRDR**: Multi-Robust Doubly-Robust

### Using Estimators Directly

```python
from cje.estimators import get_estimator

# Create estimator
estimator = get_estimator("DRCPO", k=5, seed=42)

# Use convenience method
result = estimator.estimate_from_logs(logs)

# Or step-by-step
estimator.fit(logs)
result = estimator.estimate()
```

## Judges

### Creating Judges

```python
from cje.judge import JudgeFactory

judge = JudgeFactory.create(
    provider="openai",
    model_name="gpt-4o",
    template="quick_judge"
)

# Score a single sample
score = judge.score(context="What is 2+2?", response="4")
print(f"Score: {score.mean} ± {score.std}")
```

### Judge Templates

- `quick_judge`: Fast single-score evaluation
- `detailed_judge`: Detailed feedback with score
- `pairwise_judge`: Compare two responses

## Data Loading

```python
from cje.data import load_dataset

# Load from file
dataset = load_dataset("./data/test.jsonl")

# Load from HuggingFace
dataset = load_dataset("ChatbotArena", split="test")

# Iterate over samples
for sample in dataset.itersamples():
    print(f"Context: {sample.context}")
    print(f"Response: {sample.response}")
    print(f"Reward: {sample.reward}")
```

## Command-Line Interface

```bash
# Run full experiment
cje run --cfg-path configs --cfg-name example_eval

# Run specific stages
cje log generate --config logging_config.yaml
cje judge score --config judge_config.yaml
cje estimate --estimator DRCPO --data results.jsonl

# Check cached stages
cje validate cache-status --work-dir ./outputs
```

## Advanced Features

### Custom Estimators

```python
from cje.estimators.base import Estimator

class MyEstimator(Estimator):
    def fit(self, logs, **kwargs):
        # Custom fitting logic
        pass
    
    def estimate(self):
        # Return EstimationResult
        pass
```

### Custom Judge Templates

```python
judge = JudgeFactory.create(
    provider="openai",
    model_name="gpt-4o",
    custom_template="""
    Rate this response from 1-10:
    Context: {context}
    Response: {response}
    
    Score:
    """
)
```

### Oracle Labeling

```python
from cje.oracle_labeling import add_oracle_labels

# Add oracle labels to dataset
rows_with_oracle = add_oracle_labels(
    rows,
    provider="openai",
    model_name="gpt-4o",
    fraction=0.2  # Label 20% of data
)
```