Teacher Forcing Guide
=====================

Understanding how CJE handles log probability computation across different API providers and the critical importance of consistency.

Overview
--------

**Teacher forcing** is a technique where we compute the log probability of a specific text sequence given a context, rather than generating new text. This is crucial for off-policy evaluation because we need to know:

1. **π₀(response|context)** - The probability the logging policy assigned to generating a specific response
2. **π'(response|context)** - The probability each target policy would assign to that same response

The importance weights π'/π₀ depend critically on these probabilities being computed consistently.

The API Compatibility Challenge
-------------------------------

Modern LLM APIs present a fundamental challenge for teacher forcing:

**Chat Completions API** (What most providers offer)
   - Designed for multi-turn conversations
   - Returns log probabilities only for *generated* tokens
   - Cannot compute log probabilities for arbitrary text
   - Example: ``client.chat.completions.create(...)``

**Completions API** (Legacy, but crucial for CJE)
   - Designed for text completion
   - Supports ``echo=True`` parameter to return log probabilities for input text
   - Allows true teacher forcing
   - Example: ``client.completions.create(..., echo=True)``

The Key Insight
~~~~~~~~~~~~~~~

Chat completions and completions APIs return **different log probabilities** for the same text! This is because:

1. Chat APIs add special tokens and formatting
2. System prompts are handled differently
3. Tokenization may differ between endpoints

**CJE's Solution**: Convert everything to completions-compatible format for log probability computation to ensure consistency.

CJE's Two-Pass Approach
-----------------------

For providers that support both APIs, CJE uses a two-pass approach:

.. code-block:: python

   # Pass 1: Generate response using chat completions API
   response = generate_with_chat_api(context)
   
   # Pass 2: Score that response using completions API with teacher forcing
   logp = score_with_completions_api(context, response, echo=True)

This ensures:
- Natural generation using the chat interface
- Consistent scoring across all policies using completions API

Implementation Details
----------------------

Format Conversion
~~~~~~~~~~~~~~~~~

CJE converts chat-style conversations to a flat text format for the completions API using the completions template system:

.. code-block:: python

   # Original chat format
   messages = [
       {"role": "system", "content": "You are a helpful assistant"},
       {"role": "user", "content": "What is machine learning?"}
   ]
   
   # Converted to completions format based on template
   # For Llama 3:
   prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|>..."
   
   # For Llama 4:
   prompt = "<|begin_of_text|><|header_start|>system<|header_end|>\n\nYou are a helpful assistant<|eot|>..."

**Important**: You must explicitly specify the correct ``completions_template_format`` for your model. See :doc:`../guides/completions_templates` for details.

Code Example
~~~~~~~~~~~~

Here's how CJE implements teacher forcing in ``api_policy.py``:

.. code-block:: python

   def _teacher_forcing_logprob(self, context: str, response: str) -> float:
       """Compute log probability using teacher forcing via completions API."""
       # Parse context into messages
       messages = parse_context(context, self.system_prompt, self.user_message_template)
       
       # Format as complete conversation including response
       full_prompt = self._format_conversation_with_response(messages, response)
       
       # Use completions API to get logprobs for the full sequence
       resp = self.client.completions.create(
           model=self.model_name,
           prompt=full_prompt,
           max_tokens=0,      # Don't generate - just score existing text
           temperature=0.0,   # Deterministic
           logprobs=5,
           echo=True,         # Return logprobs for input text
       )
       
       # Extract log probabilities for the response portion
       logprobs_data = resp.choices[0].logprobs
       response_logprob = sum_response_logprobs_tail(
           logprobs_data.token_logprobs,
           response_token_count
       )
       
       return response_logprob

Provider Support Status
-----------------------

As of late 2024, teacher forcing support across providers:

.. list-table:: Provider Teacher Forcing Support
   :header-rows: 1
   :widths: 20 20 60

   * - Provider
     - Completions API
     - Notes
   * - **Fireworks**
     - ✅ Confirmed
     - Full completions API with echo support
   * - **Together**
     - ⚠️ Unconfirmed
     - API exists but not thoroughly tested
   * - **OpenAI**
     - ❌ Deprecated
     - Completions API deprecated Nov 2023
   * - **Anthropic**
     - ❌ Never supported
     - Chat-only API design
   * - **Google**
     - ❌ Not supported
     - No completions-style API
   * - **HuggingFace**
     - ✅ Local models
     - Direct access to model internals

Implications for Off-Policy Evaluation
--------------------------------------

Without Teacher Forcing
~~~~~~~~~~~~~~~~~~~~~~~

For providers without completions API support, CJE must rely on generation-time log probabilities:

1. **During logging**: Store token-level log probabilities during generation
2. **For evaluation**: Use stored values as π₀
3. **Limitation**: Cannot compute π' for arbitrary target policies

With Teacher Forcing
~~~~~~~~~~~~~~~~~~~~

Providers with completions API support enable full off-policy evaluation:

1. **Flexibility**: Can evaluate any target policy post-hoc
2. **Consistency**: All policies scored with the same method
3. **Accuracy**: No approximation needed

Best Practices
--------------

1. **Prefer Fireworks** for experiments requiring teacher forcing
   
2. **Store token-level log probabilities** during initial generation:

   .. code-block:: python
   
      results = runner.generate_with_logp(
          prompts,
          return_token_logprobs=True  # Store for future use
      )

3. **Use consistent scoring** via ``generate_with_consistent_logp()``:

   .. code-block:: python
   
      # Ensures behavior policy uses same scoring as target policies
      results = runner.generate_with_consistent_logp(prompts)

4. **Monitor weight diagnostics** to detect scoring inconsistencies:
   
   - Identical policies should have weights ≈ 1.0
   - Large deviations indicate teacher forcing issues

Common Issues and Solutions
---------------------------

Issue: Weights Don't Equal 1.0 for Identical Policies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: When evaluating a policy identical to the logging policy, importance weights significantly deviate from 1.0.

**Cause**: Inconsistent log probability computation between generation and scoring.

**Solution**: Ensure both use the same scoring method:

.. code-block:: python

   # Wrong: Different methods
   behavior_logp = chat_api_generation_logp
   target_logp = completions_api_teacher_forcing_logp
   
   # Right: Same method
   behavior_logp = completions_api_teacher_forcing_logp
   target_logp = completions_api_teacher_forcing_logp

Issue: Different Tokenization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: Log probabilities vary slightly even with the same API.

**Cause**: Different tokenization between chat and completions formats.

**Solution**: Always use the same format conversion:

.. code-block:: python

   # Ensure consistent formatting
   def format_for_completions(messages, response):
       # Use model-specific template
       if "llama" in model_name:
           return llama_format(messages, response)
       elif "mixtral" in model_name:
           return mixtral_format(messages, response)
       # ... etc

Technical Details
-----------------

Log Probability Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~

When using ``echo=True``, the completions API returns log probabilities for the entire input. CJE must extract only the response portion:

.. code-block:: python

   def sum_response_logprobs_tail(all_token_logprobs, response_token_count):
       """Extract log probabilities for the last N tokens (the response)."""
       if response_token_count >= len(all_token_logprobs):
           return sum(all_token_logprobs)
       
       # Take last response_token_count tokens
       response_logprobs = all_token_logprobs[-response_token_count:]
       return sum(response_logprobs)

Token Counting
~~~~~~~~~~~~~~

Accurate token counting is critical:

.. code-block:: python

   import tiktoken
   
   def get_response_token_count(response: str, model_name: str) -> int:
       try:
           enc = tiktoken.encoding_for_model(model_name)
       except KeyError:
           enc = tiktoken.get_encoding("cl100k_base")  # Fallback
       return len(enc.encode(response))

Future Considerations
---------------------

As the LLM API landscape evolves:

1. **Completions API deprecation**: More providers may drop completions support
2. **Alternative solutions**: 
   - Structured generation with forced tokens
   - Custom inference endpoints
   - Open models with direct logit access
3. **Standardization efforts**: Push for teacher forcing in chat APIs

For now, **Fireworks remains the most reliable provider** for experiments requiring true teacher forcing capabilities.

See Also
--------

- :doc:`weight_processing` - How importance weights are computed and diagnosed
- :doc:`../api/loggers` - Policy runner implementation details
- :doc:`../theory/mathematical_foundations` - Theoretical importance of consistent scoring