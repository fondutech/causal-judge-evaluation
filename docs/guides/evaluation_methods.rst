Evaluation Methods in CJE
=========================

This guide covers all evaluation approaches in CJE: oracle validation, uncertainty quantification, trajectory analysis, and pairwise comparisons.

Oracle Evaluation
-----------------

Oracle evaluation provides ground truth validation using either stronger AI models or human annotators.

Types of Oracle
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Type
     - Automated (AI)
     - Human Crowdsourced
   * - **Speed**
     - Minutes
     - Days
   * - **Cost**
     - $0.01-0.03/label
     - $0.08-0.30/label
   * - **Quality**
     - Strong approximation
     - True human judgment
   * - **Use Case**
     - Development, validation
     - Research, production

Automated Oracle Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Enable oracle in config
   config = simple_config(
       dataset_name="./data.jsonl",
       oracle_enabled=True,
       oracle_model="gpt-4o",
       oracle_fraction=0.25  # Label 25% with oracle
   )
   
   # Or manually for custom validation
   from cje.judge import JudgeFactory
   
   oracle_judge = JudgeFactory.create("openai", model="gpt-4o")
   proxy_judge = JudgeFactory.create("openai", model="gpt-3.5-turbo")
   
   # Score subset with both judges
   oracle_scores = []
   proxy_scores = []
   for sample in dataset[:100]:
       oracle_scores.append(oracle_judge.score(sample))
       proxy_scores.append(proxy_judge.score(sample))
   
   # Check correlation
   correlation = np.corrcoef(oracle_scores, proxy_scores)[0, 1]
   print(f"Judge correlation: {correlation:.3f}")

Human Oracle Integration
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Export for human labeling
   from cje.oracle_labeling import export_for_labeling
   
   export_for_labeling(
       dataset=dataset,
       output_file="surge_task.json",
       platform="surge",
       samples_per_item=3
   )
   
   # After labeling, import results
   human_labels = load_json("surge_results.json")
   oracle_scores = [label['rating'] / 10.0 for label in human_labels]

Validation Metrics
~~~~~~~~~~~~~~~~~~

- **Absolute Error**: ``|v_hat - v_oracle|``
- **Relative Error**: ``|v_hat - v_oracle| / v_oracle``
- **CI Coverage**: Does 95% CI contain oracle truth?
- **Correlation**: Spearman ρ between proxy and oracle

Quality Thresholds:
   - 🟢 Excellent: < 5% error, > 0.8 correlation
   - 🟡 Good: < 10% error, > 0.7 correlation
   - 🔴 Poor: > 15% error, < 0.6 correlation

Uncertainty Quantification
--------------------------

CJE provides three methods for quantifying judge uncertainty:

Uncertainty Methods
~~~~~~~~~~~~~~~~~~~

**1. Deterministic (No Uncertainty)**

.. code-block:: python

   judge = JudgeFactory.create(
       provider="openai",
       model="gpt-4o",
       uncertainty_method="none"  # variance = 0
   )

**2. Confidence Intervals**

.. code-block:: python

   judge = JudgeFactory.create(
       provider="fireworks",
       model="llama-v3-70b",
       uncertainty_method="confidence_interval"
   )
   
   # Returns JudgeScoreWithCI
   score = judge.score(context, response)
   print(f"Score: {score.mean} [{score.ci_lower}, {score.ci_upper}]")
   print(f"Variance: {score.variance}")

**3. Monte Carlo Sampling**

.. code-block:: python

   judge = JudgeFactory.create(
       provider="openai",
       model="gpt-4",
       uncertainty_method="monte_carlo",
       temperature=0.3,
       mc_samples=10
   )
   
   # Samples multiple times and computes variance
   score = judge.score(context, response)
   print(f"Mean: {score.mean}, Variance: {score.variance}")

Choosing Uncertainty Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Method
     - Best For
     - Pros
     - Cons
   * - Deterministic
     - Quick tests, stable models
     - Fast, reproducible
     - No uncertainty info
   * - Confidence Interval
     - Production use
     - Single call, calibrated
     - Requires compatible model
   * - Monte Carlo
     - Research, any model
     - Works everywhere
     - Slow (N calls)

Impact on Estimators
~~~~~~~~~~~~~~~~~~~~

Uncertainty-aware judges improve estimator performance:

.. code-block:: python

   # Estimators automatically use judge variance
   estimator = get_estimator("DRCPO")
   result = estimator.estimate()
   
   # Standard errors account for judge uncertainty
   print(f"SE with uncertainty: {result.se[0]:.3f}")

Trajectory Evaluation
---------------------

For multi-turn conversations and sequential decisions.

Data Format
~~~~~~~~~~~

.. code-block:: json

   {
     "trajectory_id": "conv_123",
     "turns": [
       {
         "context": "Previous conversation...",
         "query": "What about the second option?",
         "response": "The second option offers...",
         "reward": 0.8
       }
     ]
   }

Trajectory Estimators
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cje.estimators import TrajectoryDRCPO
   
   # Handles sequential dependencies
   estimator = TrajectoryDRCPO(
       gamma=0.95,  # Discount factor
       handle_padding=True
   )
   
   # Estimates consider full trajectories
   result = estimator.estimate()

Key Differences
~~~~~~~~~~~~~~~

- **Temporal dependencies**: Later turns depend on earlier ones
- **Cumulative rewards**: Total conversation quality
- **Variable lengths**: Handle padding appropriately
- **Context accumulation**: Growing context over turns

Pairwise Evaluation
-------------------

Convert pairwise comparisons (A vs B) into absolute scores.

Bradley-Terry Model
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cje.models import BradleyTerryModel
   
   # Pairwise comparison data
   comparisons = [
       {"winner": "gpt-4", "loser": "gpt-3.5", "magnitude": 0.8},
       {"winner": "claude", "loser": "gpt-3.5", "magnitude": 0.6},
       # ...
   ]
   
   # Fit model
   bt_model = BradleyTerryModel()
   bt_model.fit(comparisons)
   
   # Get utilities
   utilities = bt_model.get_utilities()
   print(utilities)  # {"gpt-4": 0.72, "claude": 0.65, "gpt-3.5": 0.50}

Integration with CJE
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Convert pairwise to absolute rewards
   for sample in dataset:
       sample['reward'] = utilities[sample['model']]
   
   # Run standard CJE
   config = simple_config(dataset_name=dataset)
   results = config.run()

Arena-Style Evaluation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For ChatBot Arena style data
   from cje.utils import arena_to_cje_format
   
   arena_data = load_json("arena_battles.json")
   cje_data = arena_to_cje_format(
       arena_data,
       use_bradley_terry=True,
       normalize_utilities=True
   )

Combined Evaluation Pipeline
----------------------------

Here's how to combine multiple evaluation methods:

.. code-block:: python

   from cje.config.unified import ConfigurationBuilder
   from cje.judge import JudgeFactory
   
   # 1. Setup with uncertainty-aware judge
   config = (ConfigurationBuilder()
       .dataset("conversations.jsonl")
       .logging_policy("gpt-3.5-turbo")
       .add_target_policy("improved", "gpt-4")
       .judge("gpt-4o", uncertainty_method="confidence_interval")
       .estimator("DRCPO")
       .build())
   
   # 2. Enable oracle validation
   config.oracle = {
       "enabled": True,
       "provider": "anthropic",
       "model": "claude-3-opus",
       "fraction": 0.3
   }
   
   # 3. Handle trajectories if needed
   if has_multi_turn_data:
       config.estimator.trajectory_aware = True
       config.estimator.gamma = 0.95
   
   # 4. Run evaluation
   results = config.run()
   
   # 5. Access comprehensive results
   print(f"Estimate: {results['estimate']}")
   print(f"Oracle correlation: {results['oracle_metrics']['correlation']}")
   print(f"Judge uncertainty: {results['judge_metrics']['avg_variance']}")

Best Practices
--------------

**Oracle Validation**
   - Use 25-30% oracle coverage for good validation
   - Stratify sampling across score ranges
   - Check correlation before trusting results

**Uncertainty Quantification**
   - Use CI method for compatible models
   - Monte Carlo for research/validation
   - Account for uncertainty in small datasets

**Trajectory Evaluation**
   - Ensure consistent context handling
   - Choose appropriate discount factor
   - Handle variable lengths properly

**Pairwise Comparisons**
   - Need sufficient comparison coverage
   - Check for transitivity violations
   - Normalize utilities for interpretability

Quick Reference
---------------

.. list-table::
   :header-rows: 1

   * - Method
     - When to Use
     - Key Config
   * - Oracle (AI)
     - Validate estimates
     - ``oracle_enabled=True``
   * - Oracle (Human)
     - Gold standard
     - Manual process
   * - Uncertainty
     - Always recommended
     - ``uncertainty_method="confidence_interval"``
   * - Trajectories
     - Multi-turn data
     - ``trajectory_aware=True``
   * - Pairwise
     - A/B comparisons
     - Use Bradley-Terry

See Also
--------

- :doc:`comprehensive_usage` - Full usage guide
- :doc:`technical_implementation` - How these methods work internally
- :doc:`/api/estimators_consolidated` - Estimator details