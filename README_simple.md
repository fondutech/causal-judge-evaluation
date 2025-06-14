# CJE: Simple & Powerful Causal LLM Evaluation

CJE makes it easy to get unbiased, causal estimates of how your LLM changes (new prompts, models, parameters) would perform in production - without deploying them.

## Quick Start (5 minutes)

```bash
# Install
pip install cje

# Set API key
export OPENAI_API_KEY="sk-..."

# Create a config
cat > my_experiment.yaml << EOF
logging_policy:
  name: current_production
  provider: openai
  model_name: gpt-3.5-turbo

target_policies:
  - name: new_prompt
    provider: openai
    model_name: gpt-3.5-turbo
    system_prompt: "Think step by step"
    
  - name: stronger_model
    provider: openai  
    model_name: gpt-4

dataset:
  name: ChatbotArena
  sample_limit: 100
EOF

# Run evaluation
cje run my_experiment.yaml

# View results
cje show outputs/results.json
```

## What CJE Does

1. **Takes your logged data** (context, response, model used)
2. **Estimates counterfactual performance** ("What if we had used GPT-4?")
3. **Provides confidence intervals** (know when differences are real)
4. **Saves GPU time** (reuses existing responses, no re-generation)

## Core Concepts (Simplified)

### 1. Policies
A "policy" is just a configuration of your LLM system:
- Model choice (GPT-3.5 vs GPT-4)
- Prompt template  
- Temperature settings
- System prompts

### 2. Importance Weights
CJE computes how likely each response was under different policies, then uses these weights to estimate performance.

### 3. Doubly-Robust Estimation
Even if the weights or outcome model aren't perfect, CJE still gives unbiased estimates (theoretical guarantee!).

## Simple Python API

```python
from cje import CJEConfig, PolicyConfig, CJEPipeline

# Define experiment
config = CJEConfig(
    logging_policy=PolicyConfig(
        name="current", 
        provider="openai",
        model_name="gpt-3.5-turbo"
    ),
    target_policies=[
        PolicyConfig(
            name="improved",
            provider="openai", 
            model_name="gpt-4"
        )
    ]
)

# Run
pipeline = CJEPipeline(config)
results = pipeline.run()

# Results include estimates, confidence intervals, and ESS
print(results.summary())
```

## Architecture (Simplified)

```
Your Logs → Importance Weights → Judge Scores → Causal Estimate → Results
     ↓              ↓                ↓              ↓              ↓
  CSV/JSON    Policy Probs      LLM Judge      DR Estimator    Rankings + CIs
```

### Key Modules

- `cje.core` - Main pipeline and results
- `cje.config.simple` - Configuration classes  
- `cje.providers.unified` - Single interface for all LLMs
- `cje.estimators` - Statistical estimators (DRCPO, MRDR)
- `cje.cli.simple_cli` - Command-line interface

## Common Use Cases

### 1. Prompt Engineering
```yaml
target_policies:
  - name: baseline
    system_prompt: "You are a helpful assistant"
    
  - name: cot
    system_prompt: "Think step-by-step before answering"
    
  - name: expert  
    system_prompt: "You are an expert. Be concise and accurate"
```

### 2. Model Comparison
```yaml
target_policies:
  - name: gpt35
    model_name: gpt-3.5-turbo
    
  - name: gpt4
    model_name: gpt-4
    
  - name: claude
    provider: anthropic
    model_name: claude-3-sonnet-20240229
```

### 3. Parameter Tuning
```yaml
target_policies:
  - name: temp_0
    temperature: 0.0
    
  - name: temp_0.5
    temperature: 0.5
    
  - name: temp_1.0
    temperature: 1.0
```

## FAQ

**Q: How much data do I need?**
A: Start with 100-1000 examples. More data = tighter confidence intervals.

**Q: What if I don't have judge scores?**
A: CJE can use any LLM as a judge. Configure in the `judge` section.

**Q: How do I know if results are significant?**
A: Check if confidence intervals overlap. Use `cje compare before.json after.json`.

**Q: What's ESS?**
A: Effective Sample Size - percentage of data that's effectively used. Higher is better. <10% suggests distribution mismatch.

## Advanced Features

- **Oracle mode**: Use stronger model for ground truth
- **Weight calibration**: Isotonic regression for better estimates
- **Cross-fitting**: k-fold CV prevents overfitting
- **Multiple estimators**: DRCPO, MRDR, IPS

See full docs for details.

## Limitations

- Requires logged probabilities (or ability to recompute)
- Assumes no hidden confounders
- Heavy distribution shift reduces effective sample size

## Get Help

- GitHub Issues: [github.com/fondutech/cje](https://github.com/fondutech/cje)
- Documentation: [cje.readthedocs.io](https://cje.readthedocs.io)
- Email: support@fondu.ai