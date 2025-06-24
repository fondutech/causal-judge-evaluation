Uncertainty-Aware Evaluation
============================

*Complete guide to using CJE's uncertainty quantification features*

CJE treats uncertainty as a first-class citizen in evaluation. Every judge score includes both a mean and variance, enabling more robust policy comparisons and better understanding of confidence in results.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

**What is Uncertainty-Aware Evaluation?**

Traditional LLM evaluation treats judge scores as point estimates. However, judges can be uncertain about their assessments - a response might be borderline between good and great, or a judge might struggle with ambiguous cases. CJE's uncertainty-aware evaluation:

1. **Quantifies judge confidence** in each score
2. **Propagates uncertainty** through the causal estimation pipeline
3. **Improves estimate robustness** via variance-based weight shrinkage
4. **Provides richer insights** through variance decomposition

**Key Benefits:**

- More accurate confidence intervals that reflect both sampling and judge uncertainty
- Automatic down-weighting of uncertain samples to improve effective sample size (ESS)
- Calibration of judge confidence to match true uncertainty
- Detailed diagnostics showing sources of variance

Quick Start
-----------

All judges in CJE now return uncertainty estimates through the ``JudgeScore`` object:

.. code-block:: python

   from cje.judge import JudgeFactory
   
   # 1. Create a judge (all judges return JudgeScore with mean+variance)
   judge = JudgeFactory.create(
       provider="openai",
       model="gpt-4o",
       template="comprehensive_judge",
       uncertainty_method="structured"  # or "deterministic" or "monte_carlo"
   )
   
   # 2. Score samples (always returns JudgeScore objects)
   samples = [{"context": "...", "response": "..."}]
   judge_scores = judge.score_batch(samples)
   # Each score has .mean and .variance attributes
   
   # 3. Use in standard CJE pipeline
   from cje.estimators import MultiDRCPOEstimator
   from cje.loggers import MultiTargetSampler
   
   # Standard CJE estimation with built-in uncertainty handling
   sampler = MultiTargetSampler([target_policy])
   estimator = MultiDRCPOEstimator(
       sampler=sampler,
       k=5,  # cross-validation folds
       judge_runner=judge  # Variance-aware features automatically included
   )
   
   # Fit and get results with uncertainty
   estimator.fit(logs)
   result = estimator.estimate()
   
   # Results include uncertainty-adjusted confidence intervals
   print(f"Policy value: {result.v_hat[0]:.3f} ± {result.se[0]:.3f}")

Setting Up Uncertainty-Aware Judges
-----------------------------------

All judges return ``JudgeScore`` objects with uncertainty. You choose the uncertainty estimation method:

1. Deterministic (Zero Variance)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For traditional point estimates:

.. code-block:: python

   from cje.judge import JudgeFactory
   
   # Creates a judge that always returns variance=0
   judge = JudgeFactory.create(
       provider="openai",
       model="gpt-4o",
       template="comprehensive_judge",
       uncertainty_method="deterministic",
       temperature=0.0
   )
   
   score = judge.score("Context", "Response")
   # score.mean = 0.75, score.variance = 0.0

2. Structured Output with Confidence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Models estimate their own uncertainty:

.. code-block:: python

   # Judge asks model to provide confidence
   judge = JudgeFactory.create(
       provider="anthropic",
       model="claude-3-sonnet-20240620",
       template="comprehensive_judge",
       uncertainty_method="structured",
       structured_output_schema="JudgeEvaluation"  # Includes confidence field
   )
   
   score = judge.score("Context", "Response")
   # score.mean = 0.75, score.variance = 0.02 (from confidence)

3. Monte Carlo Sampling
~~~~~~~~~~~~~~~~~~~~~~~

Estimate variance through repeated sampling:

.. code-block:: python

   # Sample multiple times with temperature > 0
   judge = JudgeFactory.create(
       provider="fireworks",
       model="accounts/fireworks/models/llama-v3-70b-instruct",
       uncertainty_method="monte_carlo",
       temperature=0.3,  # Must be > 0 for variance
       mc_samples=10     # Number of samples
   )
   
   score = judge.score("Context", "Response")
   # score.mean = 0.75, score.variance = 0.03 (empirical variance)

Understanding JudgeScore
------------------------

The ``JudgeScore`` object is the foundation of uncertainty-aware evaluation:

.. code-block:: python

   from cje.judge import JudgeScore
   
   # Create a score
   score = JudgeScore(mean=0.75, variance=0.02)
   
   # Access properties
   print(f"Mean: {score.mean}")              # 0.75
   print(f"Variance: {score.variance}")      # 0.02
   print(f"Std Error: {score.se}")           # 0.141 (sqrt of variance)
   
   # Confidence intervals
   lower, upper = score.confidence_interval(alpha=0.05)
   print(f"95% CI: [{lower:.3f}, {upper:.3f}]")
   
   # Automatic normalization (0-10 to 0-1)
   score10 = JudgeScore(mean=7.5, variance=2.0)  # Automatically scaled

Variance in the Estimation Pipeline
-----------------------------------

CJE automatically incorporates judge variance throughout the pipeline:

1. **Feature Engineering**: When ``judge_runner`` is provided, variance is included as a feature
2. **Calibration**: Isotonic calibration preserves variance structure
3. **Weight Shrinkage**: High-variance samples receive lower importance weights
4. **Confidence Intervals**: Final CIs reflect both sampling and judge uncertainty

.. code-block:: python

   # Variance flows through the entire pipeline
   from cje.estimators import MultiDRCPOEstimator
   
   estimator = MultiDRCPOEstimator(
       sampler=sampler,
       k=5,
       judge_runner=judge,  # Enables variance-aware features
       calibrate_weights=True,  # Preserves variance during calibration
       calibrate_outcome=True   # Outcome model uses variance
   )
   
   # Fit includes variance information
   estimator.fit(logs)
   
   # Access detailed variance information
   print(f"Reward variances: {estimator._reward_variances_full}")

Advanced: Uncertainty Calibration
---------------------------------

Judge confidence may not match actual uncertainty. CJE provides calibration tools:

.. code-block:: python

   from cje.calibration import fit_isotonic, plot_reliability
   
   # Calibrate judge scores against ground truth
   calibrator = fit_isotonic(
       judge_scores=[s.mean for s in scores],
       true_values=oracle_labels
   )
   
   # Apply calibration while preserving variance
   calibrated_scores = []
   for score in scores:
       cal_mean = calibrator.predict([score.mean])[0]
       # Preserve original variance structure
       cal_score = JudgeScore(mean=cal_mean, variance=score.variance)
       calibrated_scores.append(cal_score)
   
   # Visualize calibration
   plot_reliability(scores, oracle_labels, save_path="calibration.png")

Best Practices
--------------

1. **Choose the Right Method**
   
   - **Deterministic**: When judge consistency is paramount
   - **Structured**: When using capable models (GPT-4, Claude)
   - **Monte Carlo**: When model confidence is unreliable

2. **Validate Uncertainty Estimates**
   
   .. code-block:: python
   
      # Check if variance aligns with disagreement
      high_var_samples = [s for s in samples if s.variance > 0.05]
      # Manually review these for ambiguity

3. **Monitor Effective Sample Size**
   
   .. code-block:: python
   
      # High variance reduces ESS
      if result.diagnostics['ess_percentage'] < 50:
          print("Warning: High uncertainty reducing effective samples")

4. **Use Variance for Active Learning**
   
   .. code-block:: python
   
      # Prioritize high-uncertainty samples for human review
      uncertain_indices = np.argsort([s.variance for s in scores])[-100:]

Common Patterns
---------------

**Pattern 1: Quick Evaluation with Uncertainty**

.. code-block:: python

   from cje.judge import JudgeFactory
   from cje.config.unified import simple_config
   
   # One-line judge creation with uncertainty
   judge = JudgeFactory.create("openai", "gpt-4o-mini")
   
   # Run full pipeline
   config = simple_config(
       dataset_name="my_data.jsonl",
       logging_model="gpt-3.5-turbo",
       logging_provider="openai",
       target_model="gpt-4",
       target_provider="openai",
       judge_model="gpt-4o-mini",
       judge_provider="openai",
       estimator_name="DRCPO"
   )
   results = config.run()

**Pattern 2: Comparing Uncertainty Methods**

.. code-block:: python

   methods = ["deterministic", "structured", "monte_carlo"]
   results = {}
   
   for method in methods:
       judge = JudgeFactory.create(
           "anthropic", "claude-3-haiku-20240307",
           uncertainty_method=method
       )
       scores = judge.score_batch(samples)
       results[method] = {
           "mean_variance": np.mean([s.variance for s in scores]),
           "variance_range": (
               min(s.variance for s in scores),
               max(s.variance for s in scores)
           )
       }

**Pattern 3: Uncertainty-Weighted Aggregation**

.. code-block:: python

   # Weight scores by inverse variance (precision)
   weights = [1 / (s.variance + 1e-6) for s in scores]
   weighted_mean = np.average(
       [s.mean for s in scores],
       weights=weights
   )

Troubleshooting
---------------

**Issue: All variances are zero**

Check your uncertainty method:

.. code-block:: python

   print(judge.config.structured_output_schema)
   # Should be JudgeEvaluation or DetailedJudgeEvaluation for structured
   
   print(judge.config.temperature)
   # Should be > 0 for monte_carlo method

**Issue: Variances seem too high/low**

Examine the distribution:

.. code-block:: python

   import matplotlib.pyplot as plt
   
   variances = [s.variance for s in scores]
   plt.hist(variances, bins=50)
   plt.xlabel("Variance")
   plt.ylabel("Count")
   plt.title("Distribution of Judge Uncertainty")

**Issue: ESS too low with uncertainty**

Consider variance shrinkage:

.. code-block:: python

   # High variance samples naturally receive lower weights
   # This improves estimate stability but reduces ESS
   # Solution: Collect more data or use more certain judges

API Reference
-------------

Key classes for uncertainty-aware evaluation:

- :class:`cje.judge.JudgeScore`: Core score object with mean and variance
- :class:`cje.judge.JudgeFactory`: Factory with uncertainty methods
- :class:`cje.judge.Judge`: Base interface returning JudgeScore
- :class:`cje.judge.MCAPIJudge`: Monte Carlo uncertainty estimation
- :class:`cje.estimators.MultiDRCPOEstimator`: Variance-aware estimation

See the full :doc:`/api/index` for detailed documentation.