Quickstart Guide  
===============

Get started with CJE in 5 minutes! This guide shows you the essential steps to run your first causal evaluation.

.. tip::
   **Not sure this is the right starting point?** Visit :doc:`start_here` to find your personalized learning path.

Basic Concepts
--------------

**What is a "Policy"?**
   In CJE, a policy is simply a specific LLM configuration. It includes:
   
   - The model (e.g., GPT-4, Claude-3)
   - The system prompt (e.g., "You are a helpful assistant")
   - Generation parameters (temperature, max tokens, etc.)
   
   The **logging policy** is what generated your existing data. The **target policies** are the new configurations you want to test.

**Off-Policy Evaluation (OPE)**
   Estimate how well a new policy would perform using historical data, without deploying it.

**Importance Sampling**
   Reweight historical data to simulate what would have happened under a different policy.

**Doubly-Robust Methods**
   Combine importance sampling with outcome models for robustness against model misspecification.

**Causal Judge Evaluation (CJE)**
   The specific methodology from the research paper that achieves **single-rate efficiency** - requiring only ONE of two models (importance weights OR outcome model) to be accurate for optimal performance.

.. note::
   **📄 Paper Connection**: The ``DRCPO`` estimator implements Algorithm 1 from the CJE paper, achieving **single-rate efficiency** through isotonic calibration. See :doc:`theory/mathematical_foundations` for full theoretical details.

30-Second Example
----------------

Here's the simplest possible example:

.. code-block:: python

   from cje.config.unified import simple_config
   
   # Run evaluation with default configuration
   config = simple_config(
       dataset_name="./data/example.jsonl",
       logging_model="gpt-3.5-turbo",
       logging_provider="openai",
       target_model="gpt-4",
       target_provider="openai",
       judge_model="gpt-4o",
       judge_provider="openai",
       estimator_name="DRCPO"
   )
   results = config.run()
   
   # Print results
   print(f"Results: {results}")
   # Results contain estimator outputs, diagnostics, and final estimates

That's it! But let's understand what's happening...

Step-by-Step Tutorial
--------------------

1. **Prepare Your Data**

First, ensure your data follows the expected format:

.. code-block:: python

   # Minimal data format (context only):
   data = [
       {"context": "What is the capital of France?"},
       {"context": "Explain machine learning"},
       {"context": "What are neural networks?"},
       # ... more examples
   ]
   
   # Or complete data (all fields provided):
   data = [
       {
           "context": "What is the capital of France?",
           "response": "The capital of France is Paris.",
           "reward": 0.9,  # Quality rating (0-1) 
           "logp": -15.2,  # Log probability under logging policy
       },
       # ... more examples
   ]

.. note::
   **Data Format**
   
   **Required**: Only ``context`` (input prompt/context string)
   
   **Optional**: CJE can automatically generate/backfill:
   
   - ``response``: Generated sequence (auto-generated from logging policy)
   - ``reward``: Numeric reward (from judge evaluation or ``y_true`` labels)  
   - ``logp``: Log probability under behavior policy (auto-computed during generation)
   
   **Backfill Commands**: ``cje backfill backfill-logp`` for missing log probabilities

2. **Configure Your Experiment**

Create a configuration file that specifies your logging policy (what generated your data) and target policies (what you want to test).

.. tip::
   For a complete configuration reference with all options and examples, see :doc:`guides/configuration_reference`.

Here's a minimal example:

.. code-block:: yaml

   # my_experiment.yaml
   dataset:
     name: "./my_data.csv"
   
   logging_policy:
     provider: "openai"
     model_name: "gpt-3.5-turbo"
   
   target_policies:
     - name: "upgraded_model"
       provider: "openai"
       model_name: "gpt-4o"
   
   judge:
     provider: "openai"
     model_name: "gpt-4o-mini"
     template: "quick_judge"
   
   estimator:
     name: "DRCPO"
     k: 5

3. **Run the Evaluation**

.. code-block:: python

   from cje.config.unified import load_config
   
   # Run complete pipeline using config file
   config = load_config("configs/my_experiment.yaml")
   results = config.run()
   
   # Or build config programmatically
   from cje.config.unified import simple_config
   config = simple_config(
       dataset_name="./my_data.csv",
       logging_model="gpt-3.5-turbo",
       logging_provider="openai",
       target_model="gpt-4o",
       target_provider="openai",
       judge_model="gpt-4o",
       judge_provider="openai",
       estimator_name="DRCPO"
   )
   results = config.run()
   
   # Access results (structure depends on estimator used)
   print("=== Evaluation Results ===")
   print(f"Results: {results}")
   
   # For programmatic access, use the estimators directly:
   from cje.estimators import get_estimator
   estimator = get_estimator("DRCPO", sampler=sampler)
   estimator.fit(data)
   estimate_result = estimator.estimate()

4. **Interpret Results**

.. code-block:: python

   # The pipeline returns a dictionary with experiment results
   print(f"Full results: {results}")
   
   # For detailed analysis, use the estimator objects directly:
   from cje.estimators import get_estimator
   
   estimator = get_estimator("DRCPO", sampler=sampler)
   estimator.fit(data)
   estimate_result = estimator.estimate()
   
   # Access estimates and diagnostics
   print(f"Estimates: {estimate_result.v_hat}")
   print(f"Standard errors: {estimate_result.se}")
   print(f"Confidence intervals: {estimate_result.confidence_interval()}")
   
   # Check diagnostics
   if hasattr(estimate_result, 'diagnostics'):
       print(f"Diagnostics: {estimate_result.diagnostics}")

Common Workflows
---------------

.. note::
   For large-scale evaluation with ChatBot Arena data, see the dedicated :doc:`guides/arena_analysis` guide.

**Choosing an Estimator**

CJE provides four estimators with different trade-offs:

- **IPS**: Fastest, simplest (good for baselines)
- **SNIPS**: Self-normalized IPS (more robust)
- **DRCPO**: Doubly-robust (recommended for most use cases)
- **MRDR**: Model-regularized (best for small samples)

See :doc:`guides/user_guide` for the complete estimator selection guide and comparison code.

**Small Sample Analysis**

For datasets with <100 samples, use bootstrap confidence intervals:

.. code-block:: python

   # Use MRDR for small samples
   estimator = get_estimator("MRDR", sampler=sampler)
   estimator.fit(data)
   result = estimator.estimate()
   
   # Get bootstrap confidence intervals
   bootstrap_ci = result.bootstrap_confidence_intervals(
       confidence_level=0.95,
       n_bootstrap=1000
   )
   
   print(f"Bootstrap CI: [{bootstrap_ci['ci_lower'][0]:.3f}, {bootstrap_ci['ci_upper'][0]:.3f}]")

**Uncertainty-Aware Evaluation**

With the unified judge system (June 2025), ALL judges include uncertainty estimates:

.. code-block:: python

   from cje.judge import JudgeFactory
   
   # Create a judge with uncertainty method
   judge = JudgeFactory.create(
       provider="openai",
       model="gpt-4o",
       uncertainty_method="structured"  # Model estimates its confidence
   )
   
   # All scores now have mean and variance
   score = judge.score("What is 2+2?", "4")
   print(f"Score: {score.mean:.2f} ± {score.variance:.3f}")

See :doc:`guides/uncertainty_evaluation` for complete details.

**Production Integration**

For production deployment, CJE provides robust error handling and caching:

.. code-block:: python

   from cje.config.unified import load_config
   
   # Configure for production workloads
   config = load_config("configs/production/production_eval.yaml")
   results = config.run()
   
   # Or build production config programmatically
   from cje.config.unified import simple_config
   config = simple_config(
       dataset_name="./data/production.jsonl",
       logging_model="gpt-3.5-turbo",
       logging_provider="openai",
       target_model="gpt-4",
       target_provider="openai",
       judge_model="gpt-4o",
       judge_provider="openai",
       estimator_name="DRCPO",
       k=10,  # More folds for production stability
       batch_size=100  # Process in batches
   )
   results = config.run()
   
   # Results include diagnostics for monitoring
   print(f"ESS health: {results.get('weight_stats', {})}")
   print(f"Calibration quality: {results.get('calibration_rmse', 'N/A')}")

Troubleshooting
--------------

For common issues and solutions, see the comprehensive :doc:`guides/troubleshooting` guide which covers:

- Configuration and validation errors
- API authentication and rate limits  
- Weight processing and ESS issues
- Uncertainty calibration problems
- Performance optimization tips

Next Steps
----------

Now that you've got the basics:

1. **Read the** :doc:`api/estimators` **guide** for detailed estimator comparison
2. **Check out** :doc:`guides/weight_processing` **for technical details**
3. **Explore** :doc:`guides/user_guide` **for advanced workflows**
4. **Join the community** on GitHub for questions and contributions

Happy evaluating! 🚀 