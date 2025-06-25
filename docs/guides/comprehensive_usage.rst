CJE Comprehensive Usage Guide
=============================

This guide covers everything you need to use CJE effectively: from quick start to advanced configuration.

Quick Start (5 Minutes)
-----------------------

**1. Install CJE**

.. code-block:: bash

   git clone https://github.com/fondutech/causal-judge-evaluation.git
   cd causal-judge-evaluation
   poetry install
   
   # Set API key
   export FIREWORKS_API_KEY="your-key-here"  # or OPENAI_API_KEY

**2. Run Your First Evaluation**

.. code-block:: python

   from cje.config.unified import simple_config
   
   # Basic evaluation
   config = simple_config(
       dataset_name="./data/test.jsonl",
       logging_model="gpt-3.5-turbo",
       target_model="gpt-4",
       judge_model="gpt-4o",
       estimator_name="DRCPO"
   )
   results = config.run()
   print(f"Target policy score: {results['results']['DRCPO']['estimates'][0]:.3f}")

**3. Command Line Usage**

.. code-block:: bash

   # Using provided config
   cje run --cfg-path configs --cfg-name example_eval
   
   # Validate config
   cje validate --cfg-path configs --cfg-name my_config

Core Concepts
-------------

**Policies**
   - **Logging policy (π₀)**: The model that generated your historical data
   - **Target policy (π')**: The new model/prompt you want to evaluate

**Estimators**
   - **IPS**: Fast, high variance
   - **SNIPS**: Better than IPS, still fast
   - **DR-CPO** (recommended): Low variance, robust
   - **MRDR**: Lowest variance, slower

**Judges**
   - AI models that score response quality
   - Can be deterministic or uncertainty-aware

Data Format
-----------

CJE expects JSONL files with specific fields:

.. code-block:: json

   {
     "prompt_id": "unique_id_123",
     "prompt": "What is the capital of France?",
     "response": "The capital of France is Paris.",
     "metadata": {"optional": "fields"}
   }

**Required fields**: ``prompt_id``, ``prompt``, ``response``

Configuration Reference
-----------------------

Basic Configuration
~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # Minimal config
   dataset:
     name: "./data/my_data.jsonl"
   
   logging_policy:
     provider: "openai"
     model_name: "gpt-3.5-turbo"
   
   target_policies:
     - name: "improved"
       provider: "openai"
       model_name: "gpt-4"
   
   estimator:
     name: "DRCPO"

Full Configuration Options
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # Complete configuration with all options
   paths:
     work_dir: "./outputs/experiment_name"
     cache_dir: "./cache"
   
   dataset:
     name: "./data/dataset.jsonl"      # Path or huggingface dataset
     split: "test"                     # For HF datasets
     max_samples: 1000                 # Limit samples (optional)
     filter_empty: true                # Remove empty responses
   
   logging_policy:
     provider: "openai"                # openai, anthropic, fireworks, together
     model_name: "gpt-3.5-turbo"
     temperature: 0.7
     max_tokens: 512
     top_p: 0.95
     api_key: "${OPENAI_API_KEY}"     # Or use environment variable
   
   target_policies:
     - name: "gpt4_helpful"
       provider: "openai"
       model_name: "gpt-4"
       temperature: 0.3
       system_prompt: "You are a helpful assistant."
     
     - name: "claude_creative"
       provider: "anthropic"
       model_name: "claude-3-sonnet-20240229"
       temperature: 0.9
   
   judge:
     provider: "openai"
     model_name: "gpt-4o"
     temperature: 0.0                  # Deterministic scoring
     template: "deterministic"         # or "confidence_interval"
     uncertainty_method: "none"        # or "confidence_interval", "monte_carlo"
   
   estimator:
     name: "DRCPO"                     # IPS, SNIPS, DRCPO, MRDR
     k: 5                              # Cross-fitting folds
     samples_per_policy: 2             # For DR estimators
     clip: 10.0                        # Weight clipping
   
   oracle:
     enabled: false                    # Use expensive oracle for validation
     provider: "openai"
     model_name: "gpt-4o"
     fraction: 0.25                    # Label 25% with oracle
   
   output:
     format: "json"                    # json, csv, both
     include_diagnostics: true
     save_raw_scores: false

Common Workflows
----------------

System Prompt Engineering
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Test different system prompts
   config = simple_config(
       dataset_name="./customer_service.jsonl",
       logging_model="gpt-3.5-turbo",
       logging_system_prompt="You are a support agent.",
       estimator_name="DRCPO"
   )
   
   # Add target policies with different prompts
   config.target_policies = [
       {
           "name": "friendly",
           "model_name": "gpt-3.5-turbo",
           "system_prompt": "You are a friendly and empathetic support agent."
       },
       {
           "name": "professional",
           "model_name": "gpt-3.5-turbo",
           "system_prompt": "You are a professional support specialist."
       },
       {
           "name": "concise",
           "model_name": "gpt-3.5-turbo",
           "system_prompt": "You are a support agent. Be very concise."
       }
   ]
   
   results = config.run()
   print(results['summary']['recommended_policy'])

Model Comparison
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Compare different models
   from cje.config.unified import multi_policy_config
   
   config = multi_policy_config(
       dataset_name="./qa_dataset.jsonl",
       logging_policy={"provider": "openai", "model_name": "gpt-3.5-turbo"},
       target_policies=[
           {"name": "gpt4", "provider": "openai", "model_name": "gpt-4"},
           {"name": "claude", "provider": "anthropic", "model_name": "claude-3-sonnet"},
           {"name": "llama", "provider": "fireworks", "model_name": "llama-v3-70b"},
       ],
       judge_model="gpt-4o",
       estimator_name="DRCPO"
   )
   
   results = config.run()
   
   # Print rankings
   for rank in results['policy_rankings']:
       print(f"{rank['policy']}: {rank['estimate']:.3f} ± {rank['se']:.3f}")

Parameter Tuning
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Test different temperature settings
   base_model = "gpt-3.5-turbo"
   temperatures = [0.0, 0.3, 0.7, 1.0]
   
   target_policies = []
   for temp in temperatures:
       target_policies.append({
           "name": f"temp_{temp}",
           "provider": "openai",
           "model_name": base_model,
           "temperature": temp
       })
   
   config = multi_policy_config(
       dataset_name="./creative_writing.jsonl",
       logging_policy={"provider": "openai", "model_name": base_model, "temperature": 0.7},
       target_policies=target_policies,
       estimator_name="SNIPS"  # Fast for parameter sweeps
   )
   
   results = config.run()

Advanced Features
-----------------

Multi-Turn Conversations
~~~~~~~~~~~~~~~~~~~~~~~~

For conversations with multiple turns:

.. code-block:: json

   {
     "prompt_id": "conv_123",
     "messages": [
       {"role": "user", "content": "Hello"},
       {"role": "assistant", "content": "Hi! How can I help?"},
       {"role": "user", "content": "What's the weather?"}
     ],
     "response": "I don't have access to current weather data."
   }

Cross-Validation
~~~~~~~~~~~~~~~~

.. code-block:: yaml

   estimator:
     name: "DRCPO"
     k: 10              # 10-fold cross-validation
     cv_stratify: true  # Stratify by score quantiles

Bootstrap Confidence Intervals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Automatically enabled for small datasets (n < 100):

.. code-block:: python

   # Results include bootstrap CIs
   if results['n_samples'] < 100:
       bootstrap_ci = results['results']['DRCPO']['bootstrap_ci']
       print(f"Bootstrap 95% CI: {bootstrap_ci}")

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Wide Confidence Intervals**
   - Cause: High variance in importance weights
   - Solution: Use DR-CPO/MRDR, collect more data, or use more similar policies

**API Rate Limits**
   - Cause: Too many concurrent requests
   - Solution: Reduce batch size in config

**Memory Issues**
   - Cause: Large datasets loaded entirely
   - Solution: Use ``max_samples`` or process in chunks

**Estimators Disagree**
   - Cause: Model misspecification or poor calibration
   - Solution: Enable oracle validation to diagnose

Quick Fixes
~~~~~~~~~~~

.. code-block:: python

   # Debug mode
   config.debug = True
   
   # Validate before running
   config.validate()
   
   # Use checkpoint for long runs
   config.checkpoint_path = "./checkpoint.pkl"
   
   # Limit samples for testing
   config.dataset.max_samples = 100

API Reference
-------------

Python API
~~~~~~~~~~

.. code-block:: python

   # Simple config
   config = simple_config(
       dataset_name="path/to/data.jsonl",
       logging_model="model-name",
       target_model="model-name",
       judge_model="judge-model",
       estimator_name="DRCPO"
   )
   
   # Multi-policy config
   config = multi_policy_config(
       dataset_name="path/to/data.jsonl",
       target_policies=[...],
       estimator_name="DRCPO"
   )
   
   # Builder pattern
   config = (ConfigurationBuilder()
       .paths("./output")
       .dataset("data.jsonl")
       .logging_policy("gpt-3.5-turbo", provider="openai")
       .add_target_policy("test", "gpt-4", provider="openai")
       .estimator("DRCPO")
       .judge("gpt-4o", provider="openai")
       .build())

CLI Commands
~~~~~~~~~~~~

.. code-block:: bash

   # Run evaluation
   cje run --cfg-path PATH --cfg-name NAME
   
   # Validate config
   cje validate --cfg-path PATH --cfg-name NAME
   
   # Show results
   cje results PATH_TO_OUTPUT
   
   # Clean checkpoints
   cje clean PATH_TO_WORKDIR

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # API Keys
   export OPENAI_API_KEY="sk-..."
   export ANTHROPIC_API_KEY="sk-ant-..."
   export FIREWORKS_API_KEY="fw_..."
   export TOGETHER_API_KEY="..."
   
   # Optional
   export CJE_CACHE_DIR="/path/to/cache"
   export CJE_LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR

Best Practices
--------------

1. **Start Simple**: Use ``simple_config()`` for initial experiments
2. **Use DR-CPO**: It's robust and works well in most cases
3. **Enable Checkpoints**: For long-running evaluations
4. **Validate First**: Always run ``config.validate()`` before ``config.run()``
5. **Monitor Diagnostics**: Check effective sample size and weight distribution
6. **Test Small**: Use ``max_samples=100`` for initial tests
7. **Version Control**: Save configs in git for reproducibility

See Also
--------

- :doc:`evaluation_methods` - Oracle validation, uncertainty, trajectories
- :doc:`technical_implementation` - How CJE works under the hood
- :doc:`/api/estimators_consolidated` - Detailed estimator reference