Arena 10K Experiments
=====================

This guide describes the production experiment pipeline in ``cje/experiments/arena_10k_simplified/``.

Overview
--------

The Arena experiments demonstrate end-to-end CJE workflow on real data:

1. **Generate responses** from multiple policies  
2. **Compute log probabilities** for importance weighting
3. **Score with judge** and optionally oracle labels
4. **Analyze with CJE** to get unbiased estimates

Quick Start
-----------

.. code-block:: bash

   cd cje/experiments/arena_10k_simplified
   
   # Set API keys
   source /path/to/set_secrets.sh
   
   # Generate data (1000 samples, all policies)
   python generate_arena_data.py --n-samples 1000
   
   # Analyze with 10% oracle coverage
   python analyze_dataset.py --data data/cje_dataset.jsonl \
                             --oracle-coverage 0.1

Pipeline Architecture
---------------------

Step 1: Generate Responses
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``generate_arena_data.py`` creates responses from configured policies:

.. code-block:: python

   # From experiment_config.py
   POLICIES = {
       "base": {
           "model": "llama-v3p3-70b-instruct",
           "system_prompt": "You are a helpful assistant."
       },
       "parallel_universe_prompt": {
           "model": "llama-v3p3-70b-instruct",  # Same model
           "system_prompt": "Imagine parallel universes..."
       },
       "premium": {
           "model": "llama-v3p1-405b-instruct",  # Bigger model
           "system_prompt": "You are a helpful assistant."
       }
   }

The script:
- Loads prompts from Arena dataset
- Generates responses from base policy
- Saves intermediate results every 20 samples
- Handles API failures gracefully

Step 2: Compute Log Probabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``compute_logprobs.py`` adds importance weights:

.. code-block:: bash

   python compute_logprobs.py --input data/responses_base.jsonl \
                              --output data/with_logprobs.jsonl

For each response, computes:
- ``base_policy_logprob``: Log P(response | prompt, base_policy)
- ``target_policy_logprobs``: Dict of log probs for each target policy

Uses teacher forcing for accurate sequence probabilities.

Step 3: Score with Judge/Oracle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``score_with_judge.py`` adds evaluation scores:

.. code-block:: bash

   # Judge scoring (fast, all samples)
   python score_with_judge.py --input data/with_logprobs.jsonl \
                              --judge gpt-4.1-nano \
                              --output data/with_judge.jsonl
   
   # Oracle scoring (slower, subset)
   python score_with_oracle.py --input data/with_judge.jsonl \
                               --oracle gpt-5 \
                               --coverage 0.1 \
                               --output data/cje_dataset.jsonl

Judge provides score for all samples; oracle labels subset for calibration.

Step 4: Analyze with CJE
~~~~~~~~~~~~~~~~~~~~~~~~

``analyze_dataset.py`` runs the full CJE pipeline:

.. code-block:: bash

   python analyze_dataset.py --data data/cje_dataset.jsonl \
                             --estimator calibrated-ips \
                             --oracle-coverage 0.1 \
                             --output-dir results/

Produces:
- Policy value estimates with confidence intervals
- Diagnostic plots and dashboards
- JSON/CSV exports of results

Configuration
-------------

All settings in ``experiment_config.py``:

.. code-block:: python

   # Models
   BASE_MODEL = "accounts/fireworks/models/llama-v3p3-70b-instruct"
   PREMIUM_MODEL = "accounts/fireworks/models/llama-v3p1-405b-instruct"
   
   # Evaluation
   EVALUATION_MODELS = {
       "judge": "gpt-4.1-nano-2025-04-14",  # Fast
       "oracle": "gpt-5-2025-08-07"         # High quality
   }
   
   # Batch sizes
   BATCH_SIZES = {
       "response_generation": 20,
       "judge_scoring": 50,
       "logprob_computation": 20
   }
   
   # Analysis settings (from ANALYSIS_CONFIG)
   ANALYSIS_CONFIG = {
       "n_folds": 5,
       "extreme_threshold_high": 100.0,
       "extreme_threshold_low": 0.01
   }

Data Management
---------------

Intermediate Files
~~~~~~~~~~~~~~~~~~

The pipeline creates several intermediate files:

.. list-table::
   :header-rows: 1

   * - File
     - Description
     - Required Fields
   * - ``responses_base.jsonl``
     - Base policy responses
     - prompt, response
   * - ``with_logprobs.jsonl``
     - + Log probabilities
     - + base_policy_logprob, target_policy_logprobs
   * - ``with_judge.jsonl``
     - + Judge scores
     - + metadata.judge_score
   * - ``cje_dataset.jsonl``
     - + Oracle labels (subset)
     - + metadata.oracle_label

Resuming Failed Runs
~~~~~~~~~~~~~~~~~~~~

All scripts support resuming:

.. code-block:: bash

   # Will skip already processed samples
   python generate_arena_data.py --n-samples 1000 --resume

Progress is saved every batch (20-50 samples).

Running Experiments
-------------------

Basic Experiment
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # 1. Generate 100 samples for testing
   python generate_arena_data.py --n-samples 100
   
   # 2. Quick analysis with default settings
   python analyze_dataset.py --data data/cje_dataset.jsonl

Full Experiment
~~~~~~~~~~~~~~~

.. code-block:: bash

   # 1. Generate 10K samples
   python generate_arena_data.py --n-samples 10000
   
   # 2. Analyze with multiple estimators
   for estimator in calibrated-ips raw-ips dr-cpo mrdr tmle; do
       python analyze_dataset.py \
           --data data/cje_dataset.jsonl \
           --estimator $estimator \
           --oracle-coverage 0.1 \
           --output-dir results/$estimator/
   done

Oracle Coverage Sweep
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Test sensitivity to oracle coverage
   for coverage in 0.05 0.1 0.2 0.5; do
       python analyze_dataset.py \
           --data data/cje_dataset.jsonl \
           --oracle-coverage $coverage \
           --output-dir results/coverage_$coverage/
   done

Custom Policies
~~~~~~~~~~~~~~~

Add new policies to ``experiment_config.py``:

.. code-block:: python

   POLICIES["my_policy"] = {
       "name": "my_policy",
       "model": BASE_MODEL,
       "temperature": 0.3,
       "system_prompt": "You are a concise assistant.",
       "description": "Low temperature for consistency"
   }

Then regenerate log probabilities for the new policy.

Monitoring and Debugging
------------------------

Check Progress
~~~~~~~~~~~~~~

.. code-block:: bash

   # Count samples generated
   wc -l data/responses_base.jsonl
   
   # Check for errors
   grep ERROR logs/generation.log
   
   # Monitor API usage
   tail -f logs/api_calls.log

Common Issues
~~~~~~~~~~~~~

**Out of Memory**:

.. code-block:: bash

   # Process in smaller batches
   python analyze_dataset.py --data data/cje_dataset.jsonl \
                             --batch-size 1000

**API Rate Limits**:

.. code-block:: python

   # Adjust batch sizes in config
   BATCH_SIZES = {
       "judge_scoring": 10,  # Smaller batches
       "oracle_scoring": 5
   }

**Missing Log Probs**:

.. code-block:: bash

   # Recompute for specific policies
   python compute_logprobs.py --input data/responses_base.jsonl \
                              --policies "premium,my_policy"

Visualization
-------------

The analysis script generates comprehensive visualizations:

.. code-block:: bash

   python analyze_dataset.py --data data/cje_dataset.jsonl \
                             --output-dir results/ \
                             --generate-plots

Creates in ``results/plots/``:
- ``weight_distributions.png``: Importance weight histograms
- ``calibration_curve.png``: Judge vs oracle calibration
- ``policy_estimates.png``: Estimates with confidence intervals
- ``diagnostics_dashboard.html``: Interactive dashboard

Results Format
--------------

JSON Output
~~~~~~~~~~~

.. code-block:: json

   {
     "estimates": [0.72, 0.68, 0.81],
     "standard_errors": [0.03, 0.04, 0.02],
     "confidence_intervals": [[0.66, 0.78], [0.60, 0.76], [0.77, 0.85]],
     "metadata": {
       "target_policies": ["clone", "parallel_universe", "premium"],
       "estimator": "calibrated-ips",
       "n_samples": 1000,
       "oracle_coverage": 0.1,
       "diagnostics": {...}
     }
   }

CSV Output
~~~~~~~~~~

.. code-block:: text

   policy,estimate,se,ci_lower,ci_upper,ess,max_weight
   clone,0.72,0.03,0.66,0.78,0.42,12.3
   parallel_universe,0.68,0.04,0.60,0.76,0.31,24.7
   premium,0.81,0.02,0.77,0.85,0.55,8.1

Best Practices
--------------

1. **Start Small**: Test with 100 samples before scaling up
2. **Monitor ESS**: Check effective sample size in diagnostics
3. **Save Intermediate**: Keep all intermediate files for debugging
4. **Use Appropriate Judge**: Fast judge for iteration, quality oracle for final
5. **Check Diagnostics**: Always review weight distributions and calibration plots

Advanced Usage
--------------

Custom Judge Prompts
~~~~~~~~~~~~~~~~~~~~

Modify ``score_with_judge.py``:

.. code-block:: python

   JUDGE_PROMPT = '''
   Evaluate this response for helpfulness and accuracy.
   Consider: clarity, completeness, correctness.
   Score from 0 (terrible) to 1 (perfect).
   '''

Fresh Draws for DR
~~~~~~~~~~~~~~~~~~~

Generate additional samples for doubly robust:

.. code-block:: bash

   python generate_fresh_draws.py \
       --dataset data/cje_dataset.jsonl \
       --policy premium \
       --draws-per-prompt 10 \
       --output data/fresh_draws_premium.jsonl

Parallel Processing
~~~~~~~~~~~~~~~~~~~

Use multiple workers:

.. code-block:: bash

   # Split data
   split -l 2500 data/prompts.jsonl data/chunk_
   
   # Process in parallel
   parallel -j 4 python generate_arena_data.py \
       --input {} --output {.}_responses.jsonl \
       ::: data/chunk_*
   
   # Combine results
   cat data/chunk_*_responses.jsonl > data/responses_base.jsonl

Next Steps
----------

- See :doc:`getting_started` for basic CJE usage
- See :doc:`estimators` for choosing estimators
- See :doc:`diagnostics` for interpreting results
- Check ``cje/experiments/arena_10k_simplified/README.md`` for latest updates