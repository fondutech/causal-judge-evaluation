Getting Started
===============

This guide walks through the basic CJE workflow for evaluating LLM improvements.

Installation
------------

.. code-block:: bash

   git clone https://github.com/fondutech/causal-judge-evaluation.git
   cd causal-judge-evaluation
   pip install -e .

Required dependencies:

- numpy
- scipy  
- scikit-learn
- pydantic

Optional for log probability computation:

- openai (for OpenAI models)
- fireworks-ai (for Fireworks models)

Basic Concepts
--------------

CJE helps you answer: **"What would happen if we deployed this new model?"**

The key insight: You can reuse historical data from your current model (base policy) 
to evaluate new models (target policies) without actually deploying them.

Core Workflow
-------------

1. **Prepare Data**: Collect prompts and responses from your base policy
2. **Compute Log Probabilities**: Calculate how likely each policy is to generate the responses  
3. **Get Judge Scores**: Use an AI judge to evaluate response quality
4. **Run CJE**: Get unbiased estimates of target policy performance

Quickest Start: Command Line
-----------------------------

The fastest way to get started is using the CLI:

.. code-block:: bash

   # Validate your data
   python -m cje validate data.jsonl
   
   # Run analysis
   python -m cje analyze data.jsonl
   
   # Save results
   python -m cje analyze data.jsonl --output results.json

The CLI will automatically handle calibration, weight computation, and estimation.

Example: High-Level API
------------------------

The simplest way to analyze a dataset programmatically:

.. code-block:: python

   from cje import analyze_dataset
   
   # One-line analysis
   results = analyze_dataset(
       "data.jsonl",
       estimator="calibrated-ips",
       oracle_coverage=0.5  # Use 50% of oracle labels
   )
   
   # Check results
   print(f"Best policy: {results.best_policy()}")
   print(f"Estimates: {results.estimates}")
   print(f"Standard errors: {results.standard_errors}")
   
   # Export results
   from cje import export_results_json
   export_results_json(results, "results.json")

Example: Detailed Workflow
---------------------------

For more control over the process:

.. code-block:: python

   from cje import (
       load_dataset_from_jsonl,
       calibrate_dataset,
       PrecomputedSampler,
       CalibratedIPS
   )
   
   # 1. Load data with log probabilities already computed
   dataset = load_dataset_from_jsonl("gpt35_responses.jsonl")
   
   # 2. Calibrate judge scores to business metrics (optional but recommended)
   calibrated_dataset, stats = calibrate_dataset(
       dataset,
       judge_field="gpt4_score",      # AI judge scores
       oracle_field="user_rating"     # Ground truth labels
   )
   
   # 3. Run CJE estimation
   sampler = PrecomputedSampler(calibrated_dataset)
   estimator = CalibratedIPS(sampler)
   results = estimator.fit_and_estimate()
   
   # 4. Analyze results
   estimates = results.estimates  # Array of estimates for each target policy
   std_errors = results.standard_errors
   
   for i, policy in enumerate(sampler.target_policies):
       print(f"{policy}: {estimates[i]:.3f} ± {std_errors[i]:.3f}")

Data Format
-----------

Your data should be in JSONL format with these required fields:

.. code-block:: json

   {
     "prompt_id": "q_001",
     "prompt": "What is machine learning?",
     "response": "Machine learning is...",
     "base_policy_logprob": -35.704,
     "target_policy_logprobs": {
       "gpt4": -32.456,
       "claude": -33.789
     },
     "metadata": {
       "judge_score": 0.85,
       "oracle_label": 0.90
     }
   }

Key fields:

- ``prompt_id``: Unique identifier for the prompt (required)
- ``base_policy_logprob``: Log probability from your current model
- ``target_policy_logprobs``: Log probabilities from models you want to evaluate
- ``metadata``: Additional fields like judge scores and oracle labels

Computing Log Probabilities
---------------------------

For Fireworks models:

.. code-block:: python

   from cje import compute_teacher_forced_logprob
   
   result = compute_teacher_forced_logprob(
       prompt="What is 2+2?",
       response="The answer is 4.",
       model="accounts/fireworks/models/llama-v3p2-3b-instruct"
   )
   
   if result.is_valid:
       print(f"Log probability: {result.value}")

Choosing an Estimator
---------------------

**CalibratedIPS** (Recommended for most cases)
   - Handles extreme weights via isotonic calibration
   - Good balance of bias and variance
   - Fast and simple

**RawIPS** (When you have lots of data)
   - Standard importance sampling
   - Unbiased but high variance
   - Use with weight clipping

**DRCPOEstimator** (When you can generate fresh samples)
   - Doubly robust with outcome modeling
   - Lower variance than IPS
   - Requires samples from target policy

**MRDREstimator** (For heterogeneous effects)
   - Policy-specific weighted outcome models
   - Best for significant distribution shifts
   - Requires cross-fitted calibration

**TMLEEstimator** (For optimal MSE)
   - Targeted minimum loss estimation
   - Best bias-variance tradeoff
   - Requires fresh draws and cross-fitting

Next Steps
----------

- See :doc:`data_format` for detailed data requirements
- See :doc:`estimators` for estimator comparison
- See :doc:`api/core` for full API reference