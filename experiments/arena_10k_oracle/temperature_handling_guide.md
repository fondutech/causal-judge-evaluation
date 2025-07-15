# Temperature Handling for Teacher Forcing

## Key Principle
When computing log probabilities for importance sampling, we must always use temperature=1.0, regardless of the temperature used during generation.

## Why?
- Temperature scaling modifies the logits: `logits_scaled = logits / temperature`
- This changes the probability distribution and thus the log probabilities
- For accurate importance weights, we need the **true** log probabilities, not temperature-scaled ones

## Implementation Strategy

### 1. During Response Generation (Step 2)
```python
# Use the policy's actual temperature
response = model.generate(
    prompt,
    temperature=policy_config["temperature"],  # e.g., 0.7, 1.0, 1.5
    max_tokens=150,
)
```

### 2. During Log Probability Computation (Step 2b)
```python
# Always use temperature=1.0 for log prob computation
log_prob = teacher_forcing.compute_log_prob(
    prompt,
    response,
    temperature=1.0,  # ALWAYS 1.0!
)
```

## Handling Temperature in LlamaCppTeacherForcing

For our fixed implementation, we have two options:

### Option 1: Temperature Correction (Recommended)
If we know the generation temperature, we can correct the log probabilities:

```python
def compute_log_prob_with_temp_correction(
    self, 
    prompt: str, 
    response: str,
    generation_temperature: float = 1.0
) -> LogProbResult:
    # Get raw log probs (computed with temp=1.0)
    result = self.compute_log_prob(prompt, response)
    
    if result.status == LogProbStatus.SUCCESS and generation_temperature != 1.0:
        # Correct for temperature scaling
        # When temp < 1: model is more confident, log probs are higher (less negative)
        # When temp > 1: model is less confident, log probs are lower (more negative)
        corrected_logprob = result.value * generation_temperature
        
        return LogProbResult(
            value=corrected_logprob,
            status=LogProbStatus.SUCCESS,
            metadata={
                **result.metadata,
                "temperature_corrected": True,
                "generation_temperature": generation_temperature,
            }
        )
    
    return result
```

### Option 2: Strict Validation
Ensure responses were generated with temperature=1.0:

```python
# In config validation
if config.uses_llama_cpp:
    for policy in [config.logging_policy] + config.target_policies:
        if policy.get("temperature", 1.0) != 1.0:
            raise ValueError(
                f"Policy {policy['name']} has temperature={policy['temperature']}. "
                "For llama.cpp teacher forcing, all policies must use temperature=1.0 "
                "during generation to ensure accurate log probabilities."
            )
```

## Practical Example

Say we have three policies:
- `pi_creative`: temperature=1.5 (more diverse)
- `pi_focused`: temperature=0.7 (more deterministic)  
- `pi_standard`: temperature=1.0 (baseline)

### Wrong Approach ❌
```python
# This gives incorrect importance weights!
logprob_creative = compute_log_prob(prompt, response, temperature=1.5)
logprob_focused = compute_log_prob(prompt, response, temperature=0.7)
```

### Correct Approach ✅
```python
# All use temperature=1.0 for log prob computation
logprob_creative = compute_log_prob(prompt, response, temperature=1.0)
logprob_focused = compute_log_prob(prompt, response, temperature=1.0)

# Then apply temperature correction if needed
logprob_creative_corrected = logprob_creative * 1.5
logprob_focused_corrected = logprob_focused * 0.7
```

## Important Notes

1. **API-based teacher forcing** (OpenAI, Anthropic) typically handles this automatically
2. **Local models** (llama.cpp) require explicit handling
3. **Validation is key**: Always verify importance weights make sense (e.g., identical policies should have weight ≈ 1.0)

## Recommended Approach for Arena 10K

For simplicity and accuracy, enforce temperature=1.0 for all policies when using llama.cpp:

```yaml
# arena_10k.yaml
logging_policy:
  name: "p0"
  temperature: 1.0  # Required for llama.cpp

target_policies:
  - name: "pi_clone"
    temperature: 1.0  # Must match p0
    
  - name: "pi_creative"
    temperature: 1.0  # Use system prompt for variety instead
    system_prompt: "Be creative and think outside the box..."
```

This avoids temperature correction complexity while maintaining evaluation accuracy.