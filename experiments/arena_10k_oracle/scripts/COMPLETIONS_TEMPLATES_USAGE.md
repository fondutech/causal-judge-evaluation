# Completions Templates Usage Guide

This guide explains how to specify completions model formats when using the CJE API policy runners for teacher forcing log probability computation.

## What are Completions Templates?

Completions templates are used to convert structured chat conversations (with roles like 'user', 'assistant', 'system') into continuous string formats required by completions API endpoints. Different models expect different formatting conventions when using the completions API with `echo=True` for consistent log probability scoring.

**Important**: These templates are specifically for use with completions API calls for teacher forcing, NOT for general prompt formatting or chat API calls.

## Available Templates

### Built-in Templates

1. **llama3** - Llama 3 style: `<s>[INST] ... [/INST] response</s>`
2. **llama4** - Llama 4 style: `<|begin_of_text|>...<|eot|>`
3. **chatml** - ChatML style: `<|im_start|>role\ncontent<|im_end|>`
4. **alpaca** - Alpaca style: `### Instruction:\n...\n### Response:\n...`

## Usage Examples

### 1. Automatic Detection (Recommended)

The system automatically detects the appropriate template based on model name and provider:

```python
from cje.loggers.api_policy import APIPolicyRunner

# Automatically uses Llama 4 template
runner = APIPolicyRunner(
    provider="fireworks",
    model_name="accounts/fireworks/models/llama4-maverick-instruct"
)

# Automatically uses Llama 3 template
runner = APIPolicyRunner(
    provider="fireworks", 
    model_name="accounts/fireworks/models/llama-v3p1-70b-instruct"
)
```

### 2. Explicit Template Format

Override auto-detection by specifying the template format:

```python
runner = APIPolicyRunner(
    provider="together",
    model_name="some-custom-model",
    template_format="chatml"  # Force ChatML format
)
```

### 3. Custom Template Implementation

Create your own template for models with unique formats:

```python
from cje.loggers.completions_templates import CompletionsTemplate, CompletionsTemplateConfig

class MyCustomTemplate(CompletionsTemplate):
    def format_with_response(self, messages, response):
        # Convert messages + response to your model's format
        prompt = ""
        for msg in messages:
            prompt += f"[{msg['role']}]: {msg['content']}\n"
        prompt += f"[assistant]: {response}"
        return prompt
    
    def format_without_response(self, messages):
        # Same format but without the response
        prompt = ""
        for msg in messages:
            prompt += f"[{msg['role']}]: {msg['content']}\n"
        prompt += "[assistant]: "
        return prompt
    
    def get_eos_token(self):
        return "<END>"

# Use the custom template
config = CompletionsTemplateConfig(custom_template=MyCustomTemplate())
runner = APIPolicyRunner(
    provider="custom",
    model_name="my-model",
    template_config=config
)
```

### 4. Configuration in YAML

For CJE experiments, specify templates in your config files:

```yaml
# config.yaml
policy_runner:
  provider: fireworks
  model_name: llama4-maverick
  template_format: llama4  # Override auto-detection if needed
  
# Or with advanced config
policy_runner:
  provider: custom
  model_name: my-model
  template_config:
    template_format: alpaca
    # Future: custom token overrides
```

## Provider Defaults

The system has smart defaults for common providers:

- **Fireworks**: Auto-detects Llama 3 vs Llama 4 models
- **Together**: Defaults to Llama 3 format, detects Alpaca models
- **OpenAI**: Uses ChatML format
- **Anthropic**: Uses Llama 3 format

## Registering Global Templates

You can also register templates globally for reuse:

```python
from cje.loggers.completions_templates import register_completions_template

# Register once
register_completions_template("myformat", MyCustomTemplate())

# Use anywhere
runner = APIPolicyRunner(
    provider="any",
    model_name="any-model",
    template_format="myformat"
)
```

## Debugging Templates

To see which template is being used:

```python
runner = APIPolicyRunner(...)
print(f"Using template: {runner.template.__class__.__name__}")

# Test formatting
messages = [{"role": "user", "content": "Hello"}]
formatted = runner._format_conversation_with_response(messages, "Hi")
print(f"Formatted: {formatted}")
```

## Common Issues

1. **Wrong log probabilities**: Usually indicates wrong template format. Check if your model needs Llama 3 vs Llama 4 format.

2. **Token extraction errors**: The template's EOS token must match what the model actually generates.

3. **API errors**: Some providers only support specific models with completions API (e.g., Fireworks, Together).