#!/usr/bin/env python3
"""Minimal debug of the positive log prob issue"""

from llama_cpp import Llama

# Create model with correct settings
model = Llama(
    model_path="/Users/eddielandesberg/PycharmProjects/causal-judge-evaluation/models/Llama-3.2-3B-Instruct-Q6_K.gguf",
    n_ctx=512,
    verbose=False,
    logits_all=True,
    seed=42,
)

# Test case from the actual data
prompt = "User: Freeway vs Fanta orange - tell me about both\n\nAssistant:"
response = "A classic debate!"
full_text = prompt + response

print(f"Prompt: {repr(prompt)}")
print(f"Response: {repr(response)}")
print(f"Full text: {repr(full_text)}")

# Following the field guide exactly
result = model.create_completion(
    prompt,
    max_tokens=0,
    echo=True,
    logprobs=1,
    temperature=1.0,  # Critical!
)

print(f"\nPrompt completion result:")
print(f"  Text returned: {repr(result['choices'][0]['text'])}")
print(f"  Usage: {result['usage']}")

if "logprobs" in result["choices"][0]:
    lp = result["choices"][0]["logprobs"]
    print(f"\nPrompt logprobs:")
    print(f"  Tokens: {lp['tokens']}")
    print(f"  Token logprobs: {lp['token_logprobs']}")

    # Sum non-None values
    valid_lps = [x for x in lp["token_logprobs"] if x is not None]
    print(f"  Sum of valid logprobs: {sum(valid_lps)}")

# Now full text
result2 = model.create_completion(
    full_text,
    max_tokens=0,
    echo=True,
    logprobs=1,
    temperature=1.0,
)

print(f"\n\nFull text completion result:")
print(f"  Text returned: {repr(result2['choices'][0]['text'])}")
print(f"  Usage: {result2['usage']}")

if "logprobs" in result2["choices"][0]:
    lp2 = result2["choices"][0]["logprobs"]
    print(f"\nFull text logprobs:")
    print(f"  Tokens: {lp2['tokens']}")
    print(f"  Token logprobs: {lp2['token_logprobs']}")

    # Sum non-None values
    valid_lps2 = [x for x in lp2["token_logprobs"] if x is not None]
    print(f"  Sum of valid logprobs: {sum(valid_lps2)}")

    # Conditional
    prompt_sum = sum(valid_lps)
    full_sum = sum(valid_lps2)
    conditional = full_sum - prompt_sum
    print(f"\n\nConditional log prob: {full_sum} - {prompt_sum} = {conditional}")
