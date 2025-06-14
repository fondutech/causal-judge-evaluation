Quickstart Guide  
===============

Get started with CJE in 5 minutes! This guide shows you the essential steps to run your first causal evaluation.

Basic Concepts
--------------

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

   from cje.pipeline import run_pipeline
   
   # Run evaluation with default configuration
   results = run_pipeline(cfg_path="configs", cfg_name="arena_test")
   
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

2. **Configure Target Policies**

Create a YAML configuration file (e.g., ``my_experiment.yaml``):

.. code-block:: yaml

   # Dataset configuration
   dataset:
     name: "ChatbotArena"     # Built-in dataset or path to your data
     split: "train"           # Dataset split to use
     sample_limit: 1000       # Optional: limit samples for testing

   # Logging policy (what generated the historical data)
   logging_policy:
     provider: "openai"
     model_name: "gpt-3.5-turbo"
     temperature: 0.7
     max_new_tokens: 512

   # Target policies (what we want to evaluate)
   target_policies:
     - name: "gpt4_upgrade"
       provider: "openai"
       model_name: "gpt-4o"
       temperature: 0.7
       mc_samples: 5          # Monte Carlo samples per context
       
     - name: "claude3"
       provider: "anthropic" 
       model_name: "claude-3-sonnet-20240229"
       temperature: 0.7
       mc_samples: 5

   # Judge configuration (for evaluating response quality)
   judge:
     provider: "openai"
     model_name: "gpt-4o-mini"
     template: "quick_judge"
     temperature: 0.0        # Deterministic for consistency

   # Estimator configuration
   estimator:
     name: "DRCPO"           # Doubly-robust (recommended)
     k: 5                    # Cross-validation folds
     clip: 20.0              # Importance weight clipping

3. **Run the Evaluation**

.. code-block:: python

   from cje.pipeline import run_pipeline
   
   # Run complete pipeline (requires Hydra config files)
   results = run_pipeline(
       cfg_path="configs",
       cfg_name="my_experiment"
   )
   
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

Arena Analysis Example
---------------------

For ChatBot Arena-style evaluation:

.. code-block:: python

   from examples.arena_interactive import ArenaAnalyzer
   
   # Initialize analyzer
   analyzer = ArenaAnalyzer()
   
   # Quick test with sample data
   analyzer.quick_test()
   
   # If you have real results
   analyzer.load_results("path/to/experiment/results.json")
   
   # Generate comprehensive analysis
   analyzer.full_analysis()
   
   # Create visualizations
   analyzer.plot_estimates()
   analyzer.plot_weight_diagnostics()

Common Workflows
---------------

**Comparing Multiple Estimators**

.. code-block:: python

   from cje.estimators import get_estimator
   from cje.loggers.multi_target_sampler import make_multi_sampler
   
   # Set up sampler
   sampler = make_multi_sampler(target_policies_config)
   
   # Compare estimators
   estimators = ["IPS", "SNIPS", "DRCPO", "MRDR"]
   results = {}
   
   for est_name in estimators:
       estimator = get_estimator(est_name, sampler=sampler)
       estimator.fit(data)
       results[est_name] = estimator.estimate()
   
   # Compare results
   for name, result in results.items():
       print(f"{name}: {result.v_hat[0]:.3f} ± {result.se[0]:.3f}")

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

**Production Integration**

For production deployment, CJE provides robust error handling and caching:

.. code-block:: python

   from cje.pipeline import run_pipeline
   
   # Configure for production workloads
   results = run_pipeline(
       cfg_path="configs/production",
       cfg_name="production_eval"
   )
   
   # Results include diagnostics for monitoring
   print(f"ESS health: {results.get('weight_stats', {})}")
   print(f"Calibration quality: {results.get('calibration_rmse', 'N/A')}")

Troubleshooting
--------------

**High Variance Results**
   - Try DR-CPO or MRDR instead of IPS
   - Increase sample size
   - Check for distribution shift

**Low Effective Sample Size (ESS)**
   - Indicates most weight concentrated on few examples
   - Use DR methods which are less sensitive
   - Check teacher forcing consistency (see :doc:`guides/teacher_forcing`)

**Model Convergence Issues**
   - Reduce model complexity
   - Increase cross-validation folds
   - Check data quality

**API Rate Limits**
   - Add delays between requests
   - Use batch processing
   - Consider local models

Next Steps
----------

Now that you've got the basics:

1. **Read the** :doc:`api/estimators` **guide** for detailed estimator comparison
2. **Check out** :doc:`guides/weight_processing` **for technical details**
3. **Browse** :doc:`examples/index` **for more advanced use cases**
4. **Join the community** on GitHub for questions and contributions

Happy evaluating! 🚀 