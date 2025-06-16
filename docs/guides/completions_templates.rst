Completions Templates Guide
===========================

This guide explains how to configure the correct completions template format for teacher forcing in CJE.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

When using teacher forcing for log probability computation, CJE must convert chat-style conversations into the continuous string format expected by completions API endpoints. Different models use different formatting conventions, and **you must explicitly specify the correct format** for your model.

.. warning::
   There is no auto-detection. Using the wrong template format will result in extremely low log probabilities and incorrect importance weights.

Quick Start
-----------

Specify the ``completions_template_format`` when configuring policies:

**In Python:**

.. code-block:: python

   from cje.loggers.api_policy import APIPolicyRunner

   # For Llama 3.x models
   runner = APIPolicyRunner(
       provider="fireworks",
       model_name="accounts/fireworks/models/llama-v3p3-70b-instruct",
       completions_template_format="llama3"  # Required!
   )

   # For Llama 4 models
   runner = APIPolicyRunner(
       provider="fireworks",
       model_name="accounts/fireworks/models/llama4-maverick-instruct",
       completions_template_format="llama4"  # Required!
   )

**In YAML Configuration:**

.. code-block:: yaml

   logging_policy:
     provider: "fireworks"
     model_name: "accounts/fireworks/models/llama-v3p3-70b-instruct"
     temperature: 0.7
     completions_template_format: "llama3"  # Required!

   target_policies:
     - name: "llama4_model"
       provider: "fireworks"
       model_name: "accounts/fireworks/models/llama4-maverick-instruct"
       temperature: 0.7
       completions_template_format: "llama4"  # Required!

Available Templates
-------------------

Llama 3 Template (``llama3``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use for:** Llama 3.0, 3.1, 3.2, 3.3 models

**Format:**

.. code-block:: text

   <|begin_of_text|>
   <|start_header_id|>user<|end_header_id|>
   
   {user_message}<|eot_id|>
   <|start_header_id|>assistant<|end_header_id|>
   
   {response}<|eot_id|>

Llama 4 Template (``llama4``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use for:** Llama 4 models (Scout, Maverick, etc.)

**Format:**

.. code-block:: text

   <|begin_of_text|>
   <|header_start|>user<|header_end|>
   
   {user_message}<|eot|>
   <|header_start|>assistant<|header_end|>
   
   {response}<|eot|>

Provider Compatibility
----------------------

.. list-table:: Provider Support for Teacher Forcing
   :header-rows: 1
   :widths: 20 30 50

   * - Provider
     - Supported Models
     - Notes
   * - **Fireworks**
     - ✅ Llama 3.x, Llama 4
     - Full completions API support with echo=True
   * - **Together**
     - ✅ Llama 3.x only
     - Llama 4 returns "Echo not yet supported"
   * - **OpenAI**
     - ❌ None
     - Completions API deprecated
   * - **Anthropic**
     - ❌ None
     - No completions API

Validation
----------

Always validate your configuration before running experiments:

.. code-block:: python

   # Validate template configuration
   runner.validate_teacher_forcing()

This will test known high-probability responses and alert you if the template is misconfigured.

Common Issues
-------------

Wrong Template Format
~~~~~~~~~~~~~~~~~~~~~

**Symptom:** Extremely low log probabilities (-20 to -25) for simple responses

**Solution:** Ensure you're using the correct template:

.. code-block:: python

   # ❌ WRONG - Using llama4 template for Llama 3 model
   runner = APIPolicyRunner(
       provider="fireworks",
       model_name="llama-v3p3-70b-instruct",
       completions_template_format="llama4"  # Wrong!
   )

   # ✅ CORRECT
   runner = APIPolicyRunner(
       provider="fireworks",
       model_name="llama-v3p3-70b-instruct",
       completions_template_format="llama3"  # Correct!
   )

Provider Incompatibility
~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom:** "Echo not yet supported for this model" error

**Solution:** Check provider compatibility:

.. code-block:: python

   # ❌ WRONG - Llama 4 on Together
   runner = APIPolicyRunner(
       provider="together",
       model_name="llama4-model",
       completions_template_format="llama4"
   )

   # ✅ CORRECT - Use Llama 3.x on Together
   runner = APIPolicyRunner(
       provider="together",
       model_name="llama-3.3-70b-instruct",
       completions_template_format="llama3"
   )

Custom Templates
----------------

For models with different formatting requirements, implement a custom template:

.. code-block:: python

   from cje.loggers.completions_templates import CompletionsTemplate, register_completions_template

   class MyCustomTemplate(CompletionsTemplate):
       def format_with_response(self, messages, response):
           # Convert messages + response to your model's format
           user_msg = next(m['content'] for m in messages if m['role'] == 'user')
           return f"User: {user_msg}\nAssistant: {response}"
       
       def format_without_response(self, messages):
           user_msg = next(m['content'] for m in messages if m['role'] == 'user')
           return f"User: {user_msg}\nAssistant: "
       
       def get_eos_token(self):
           return "\n"

   # Register the template
   register_completions_template("mycustom", MyCustomTemplate())

   # Use it
   runner = APIPolicyRunner(
       provider="myprovider",
       model_name="mymodel",
       completions_template_format="mycustom"
   )

Complete Example
----------------

Here's a complete example showing proper configuration:

.. code-block:: yaml

   # config/experiment.yaml
   dataset:
     name: "ChatbotArena"
     split: "train"
     sample_limit: 100

   logging_policy:
     provider: "fireworks"
     model_name: "accounts/fireworks/models/llama-v3p3-70b-instruct"
     temperature: 0.7
     max_new_tokens: 1000
     completions_template_format: "llama3"  # Critical!

   target_policies:
     - name: "llama3_variant"
       provider: "fireworks"
       model_name: "accounts/fireworks/models/llama-v3p1-70b-instruct"
       temperature: 0.5
       completions_template_format: "llama3"
       mc_samples: 5
     
     - name: "llama4_model"
       provider: "fireworks"
       model_name: "accounts/fireworks/models/llama4-scout-instruct"
       temperature: 0.7
       completions_template_format: "llama4"
       mc_samples: 5

   judge:
     provider: "openai"
     model_name: "gpt-4o-mini"
     template: "quick_judge"

   estimator:
     name: "DRCPO"
     k: 5

Key Takeaways
-------------

1. **Always specify** ``completions_template_format`` explicitly
2. **Validate early** with ``validate_teacher_forcing()``
3. **Check compatibility** - not all providers support all models
4. **Use correct format** - ``llama3`` for Llama 3.x, ``llama4`` for Llama 4
5. **Monitor diagnostics** - watch for suspiciously low log probabilities

See Also
--------

- :doc:`teacher_forcing` - Technical details about teacher forcing
- :doc:`weight_processing` - How importance weights are computed
- :doc:`configuration_reference` - Full configuration options