# Completions Templates Usage Guide

This guide explains how to specify completions model formats when using the CJE API policy runners for teacher forcing log probability computation.

## What are Completions Templates?

Completions templates are used to convert structured chat conversations (with roles like 'user', 'assistant', 'system') into continuous string formats required by completions API endpoints. Different models expect different formatting conventions when using the completions API with `echo=True` for consistent log probability scoring.

**Important**: These templates are specifically for use with completions API calls for teacher forcing, NOT for general prompt formatting or chat API calls.

**Provider Support**: Currently, only Fireworks is confirmed to support the completions API with echo=True required for teacher forcing. While the CJE codebase includes adapters for other providers (OpenAI, Anthropic, Together, etc.), teacher forcing functionality is only available with Fireworks.

## Available Templates

### Built-in Template

Currently, only the **llama4** template is provided out of the box:
- **llama4** - Llama 4 style: `<|begin_of_text|><|header_start|>role<|header_end|>\n\n{content}<|eot|>`

This template is specifically designed for Llama 4 models (Scout, Maverick, etc.) and is the default for all providers.

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
```

### 2. Explicit Template Format

Override auto-detection by specifying the template format:

```python
runner = APIPolicyRunner(
    provider="fireworks",
    model_name="some-custom-model",
    template_format="llama4"  # Explicitly specify llama4 (though it's the default)
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
  # template_format: llama4  # Not needed, it's the default
  
# Or with custom template
policy_runner:
  provider: custom
  model_name: my-model
  template_config:
    custom_template: !MyCustomTemplate {}
```

## Provider Support

Currently, only **Fireworks** is confirmed to support the teacher forcing completions API with echo=True:

- **Fireworks**: Uses Llama 4 format (confirmed working)
- **Other providers**: Not yet supported for teacher forcing

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

1. **Wrong log probabilities**: If you're using a non-Llama-4 model, you'll need to implement a custom template.

2. **Token extraction errors**: The template's EOS token (`<|eot|>`) must match what the model actually generates.

3. **API errors**: Currently only Fireworks supports the completions API with echo=True for teacher forcing.

## Adding Support for Other Models

If you need to support models other than Llama 4, you'll need to implement a custom template following the `CompletionsTemplate` interface. See the Llama 4 implementation in `cje/loggers/completions_templates.py` as a reference.