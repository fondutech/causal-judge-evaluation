#!/usr/bin/env python3
"""Debug token issue with max_tokens=0"""

from llama_cpp import Llama

# Create model
model = Llama(
    model_path="/Users/eddielandesberg/PycharmProjects/causal-judge-evaluation/models/Llama-3.2-3B-Instruct-Q6_K.gguf",
    n_ctx=256,
    n_gpu_layers=0,
    verbose=False,
    logits_all=True,
    seed=42,
)

# Test text
text = "Hello world"
print(f"Input text: '{text}'")

# Try different approaches
print("\n1. With max_tokens=0, echo=True (field guide recipe):")
result = model.create_completion(
    text,
    max_tokens=0,
    echo=True,
    logprobs=1,
    temperature=1.0,
)
print(f"  Generated: {result['usage']['completion_tokens']} tokens")
print(f"  Text length: {len(result['choices'][0]['text'])} chars")

# Check the tokens
if "logprobs" in result["choices"][0]:
    lp = result["choices"][0]["logprobs"]
    tokens = lp.get("tokens", [])
    print(f"  First 10 tokens: {tokens[:10]}")

    # Find where "Hello world" ends
    for i, token in enumerate(tokens[:20]):
        print(f"    {i}: '{token}'")

print("\n2. Try with echo=False:")
result2 = model.create_completion(
    text,
    max_tokens=0,
    echo=False,
    logprobs=1,
    temperature=1.0,
)
print(f"  Generated: {result2['usage']['completion_tokens']} tokens")
print(f"  Text: '{result2['choices'][0]['text'][:50]}...'")

print("\n3. Try tokenize + eval approach:")
tokens = model.tokenize(text.encode())
print(f"  Tokenized '{text}' -> {tokens}")

# Use eval to process tokens without generation
model.reset()
model.eval(tokens)
print(f"  Evaluated {len(tokens)} tokens")

# Can we get log probs this way?
print("\n4. Direct tokenization check:")
# Include BOS token explicitly
bos = model.token_bos()
print(f"  BOS token: {bos}")
print(f"  Our tokens include BOS: {bos in tokens}")
