Estimators API Reference
========================

CJE provides several off-policy evaluation estimators, each with different bias-variance trade-offs. This page consolidates all estimator documentation including API reference, DR requirements, and usage guidelines.

Overview
--------

All estimators inherit from the base :class:`~cje.estimators.base.Estimator` class and return 
:class:`~cje.estimators.results.EstimationResult` objects with standardized interfaces.

.. autoclass:: cje.estimators.base.Estimator
   :members:

.. autoclass:: cje.estimators.results.EstimationResult
   :members:

Quick Selection Guide
---------------------

.. list-table:: Estimator Selection at a Glance
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Estimator
     - Best For
     - Speed
     - Variance
     - Requirements
   * - **IPS**
     - Large datasets, quick tests
     - ⚡⚡⚡ Fastest
     - ❌ High
     - None
   * - **SNIPS**
     - Medium datasets, better than IPS
     - ⚡⚡⚡ Fast
     - ✅ Lower
     - None
   * - **DR-CPO**
     - **Most use cases** (recommended)
     - ⚡⚡ Medium
     - ✅✅ Much lower
     - Target samples
   * - **MRDR**
     - Small datasets, max precision
     - ⚡ Slow
     - ✅✅✅ Lowest
     - Target samples

Inverse Propensity Scoring (IPS)
--------------------------------

The simplest importance sampling estimator.

**Mathematical Foundation:**

.. math::

   \hat{V}^{\text{IPS}}(\pi) = \frac{1}{n} \sum_{i=1}^n w_i r_i

where :math:`w_i = \frac{\pi(a_i|x_i)}{\pi_0(a_i|x_i)}` are importance weights.

**Characteristics:**

- ✅ Unbiased under correct propensity estimates
- ✅ No target samples needed
- ❌ High variance with distribution shift
- ⚡ Very fast computation

.. autoclass:: cje.estimators.ips_only_estimators.MultiIPSEstimator
   :members:
   :show-inheritance:

**When to Use:**

- Large sample sizes (n > 1000)
- Small distribution shift between policies
- Need fastest possible computation
- Quick prototyping and testing

Self-Normalized IPS (SNIPS)
---------------------------

Reduces variance by normalizing importance weights.

**Mathematical Foundation:**

.. math::

   \hat{V}^{\text{SNIPS}}(\pi) = \frac{\sum_{i=1}^n w_i r_i}{\sum_{i=1}^n w_i}

**Characteristics:**

- ✅ Lower variance than IPS
- ✅ No target samples needed
- ❌ Introduces small bias
- ⚡ Fast computation

.. autoclass:: cje.estimators.ips_only_estimators.MultiSNIPSEstimator
   :members:
   :show-inheritance:

**When to Use:**

- Medium sample sizes (100 < n < 1000)
- Moderate distribution shift
- Want variance reduction over IPS without target sampling
- Standard off-policy evaluation tasks

Calibrated IPS
--------------

Improves IPS by calibrating importance weights to reduce bias.

**Mathematical Foundation:**

Uses isotonic regression to calibrate weights:

.. math::

   \hat{w}_i = g(w_i)

where :math:`g` is learned via isotonic regression on validation data.

**Characteristics:**

- ✅ Lower bias than standard IPS
- ✅ Handles weight misspecification
- ✅ No target samples needed
- ⚡ Fast after calibration

.. autoclass:: cje.estimators.ips_only_estimators.MultiCalibratedIPSEstimator
   :members:
   :show-inheritance:

**When to Use:**

- Suspect importance weight misspecification
- Have validation data with ground truth
- Want robustness without target sampling

Doubly-Robust CPO (DR-CPO)
--------------------------

Combines importance sampling with outcome modeling for robustness. **This is the recommended estimator for most use cases.**

**Mathematical Foundation:**

.. math::

   \hat{V}^{\text{DR}}(\pi) = \frac{1}{n} \sum_{i=1}^n \left[ \mu_\pi(x_i) + w_i(r_i - \mu(x_i, a_i)) \right]

where:

- :math:`\mu_\pi(x_i) = \mathbb{E}_{s \sim \pi}[\mu(x_i, s)]` is the baseline term
- :math:`\mu(x, a)` is the outcome model predicting reward

**Characteristics:**

- ✅ Robust to model misspecification (doubly robust)
- ✅ Much lower variance than IPS/SNIPS
- ✅ Unbiased if either weights OR outcome model is correct
- ⚠️ **Requires target policy samples** for variance reduction

.. autoclass:: cje.estimators.doubly_robust_estimators.MultiDRCPOEstimator
   :members:
   :show-inheritance:

DR Requirements: Target Samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::
   DR-CPO requires target policy samples to achieve variance reduction. Without them (``samples_per_policy=0``), it reduces to standard IPW with no benefit.

**Why Target Samples Are Needed:**

The baseline term :math:`\mu_\pi(x) = \mathbb{E}_{s \sim \pi}[\mu(x, s)]` requires sampling from the target policy. This term:

1. Provides the variance reduction benefit of DR
2. Cannot be computed from logged data alone
3. Requires generating new responses from target policies

**Setting samples_per_policy:**

.. code-block:: python

   # Recommended: Generate 2 samples per context
   estimator = MultiDRCPOEstimator(
       sampler=sampler,
       samples_per_policy=2,  # ✅ Variance reduction enabled
       score_target_policy_sampled_completions=True
   )

   # Not recommended: No target samples
   estimator = MultiDRCPOEstimator(
       sampler=sampler,
       samples_per_policy=0,  # ❌ No variance reduction!
   )

**What Happens During Fitting:**

1. For each context, generate ``samples_per_policy`` responses from each target policy
2. Score them with the judge (if enabled)
3. Use them to estimate the baseline term
4. Samples are logged to ``work_dir/target_samples.jsonl``

**When to Use DR-CPO:**

- Any sample size (robust across scenarios)
- Moderate to large distribution shift
- Want robustness to model misspecification
- Can afford target sample generation
- **Recommended default choice**

Model-Regularized DR (MRDR)
---------------------------

Advanced doubly-robust method with variance-optimized outcome models.

**Mathematical Foundation:**

.. math::

   \hat{V}^{\text{MRDR}}(\pi) = \frac{1}{n} \sum_{i=1}^n \left[ \mu_\pi^*(x_i) + w_i(r_i - \mu^*(x_i, a_i)) \right]

where :math:`\mu^*` is chosen to minimize asymptotic variance.

**Characteristics:**

- ✅ Lowest variance among all estimators
- ✅ Semiparametrically efficient
- ✅ Optimal for small samples
- ⚠️ **Requires target policy samples**
- ❌ Computationally intensive

.. autoclass:: cje.estimators.doubly_robust_estimators.MultiMRDREstimator
   :members:
   :show-inheritance:

**When to Use:**

- Small sample sizes (n < 100)
- Large distribution shift
- Need maximum precision
- Can afford computational cost
- Critical decision-making scenarios

Usage Examples
--------------

Basic Estimation
~~~~~~~~~~~~~~~~

.. code-block:: python

   from cje.estimators import get_estimator
   from cje.loggers.multi_target_sampler import make_multi_sampler
   
   # Create sampler for target policies
   sampler = make_multi_sampler(target_policies_config)
   
   # Initialize estimator
   estimator = get_estimator("DRCPO", sampler=sampler, k=5)
   
   # Fit and estimate
   estimator.fit(logs)
   result = estimator.estimate()
   
   # Access results
   print(f"Estimates: {result.v_hat}")
   print(f"Standard errors: {result.se}")
   print(f"Best policy: {result.best_policy()}")

Comparing Estimators
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   estimators = ["IPS", "SNIPS", "DRCPO", "MRDR"]
   results = {}
   
   for est_name in estimators:
       estimator = get_estimator(est_name, sampler=sampler)
       estimator.fit(logs)
       results[est_name] = estimator.estimate()
   
   # Compare results
   for name, result in results.items():
       print(f"{name}: {result.v_hat[0]:.3f} ± {result.se[0]:.3f}")

EstimationResult Methods
~~~~~~~~~~~~~~~~~~~~~~~~

The EstimationResult class provides rich analysis methods:

.. code-block:: python

   # Get confidence intervals
   ci_lower, ci_upper = result.confidence_interval(level=0.95)
   print(f"95% CI: [{ci_lower[0]:.3f}, {ci_upper[0]:.3f}]")
   
   # Rank policies from best to worst
   rankings = result.rank_policies()
   for rank, (idx, score) in enumerate(rankings):
       print(f"{rank+1}. Policy {idx}: {score:.3f}")
   
   # Compare two specific policies
   comparison = result.compare_policies(0, 1)
   print(f"Policy 0 vs 1: diff = {comparison['diff']:.3f}, p = {comparison['p_value']:.3f}")
   
   # Get human-readable summary
   print(result.summary())
   
   # Convert to dictionary for serialization
   result_dict = result.to_dict()

Bootstrap Confidence Intervals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For small samples, bootstrap CIs are automatically computed:

.. code-block:: python

   result = estimator.estimate()
   
   if result.n < 100:
       # Bootstrap CIs available in metadata
       bootstrap_cis = result.metadata.get('bootstrap_confidence_intervals')
       if bootstrap_cis:
           print(f"Bootstrap 95% CI: {bootstrap_cis['95%']}")

Common Issues and Solutions
---------------------------

High Variance Estimates
~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:** Very wide confidence intervals, unstable estimates

**Solutions:**

1. Check effective sample size:

   .. code-block:: python

      ess_pct = result.metadata.get('ess_percentage', 100)
      if ess_pct < 10:
          print("⚠️ Low effective sample size!")

2. Switch to variance-reducing estimator:

   .. code-block:: python

      # Instead of IPS
      estimator = get_estimator("DRCPO", samples_per_policy=2)

3. Reduce distribution shift or collect more data

Failed Convergence
~~~~~~~~~~~~~~~~~~

**Symptoms:** DR estimators fail to fit outcome model

**Solutions:**

1. Use simpler outcome model:

   .. code-block:: python

      from sklearn.linear_model import Ridge
      
      estimator = get_estimator("DRCPO",
                              outcome_model_cls=Ridge,
                              outcome_model_kwargs={'alpha': 1.0})

2. Fall back to IPS/SNIPS:

   .. code-block:: python

      estimator = get_estimator("SNIPS", clip=10.0)

No Target Sample Budget
~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:** Cannot generate target policy samples

**Solutions:**

1. Use IPS-only estimators:

   .. code-block:: python

      # These don't need target samples
      estimator = get_estimator("SNIPS")  # or "IPS", "CalibratedIPS"

2. Pre-generate and cache samples:

   .. code-block:: python

      # Generate once, reuse multiple times
      sampler.generate_and_cache_samples(contexts, cache_file="target_samples.pkl")

Theoretical Properties
----------------------

**Consistency:** All estimators are consistent under their respective assumptions

**Efficiency:** 

- IPS: √n-consistent but not efficient
- SNIPS: √n-consistent, lower variance than IPS
- DR-CPO: Single-rate efficient (Paper's Algorithm 1)
- MRDR: Semiparametrically efficient (optimal)

**Robustness:**

- IPS/SNIPS: Require correct importance weights
- DR-CPO/MRDR: Doubly robust - consistent if EITHER weights OR outcome model is correct

**Finite Sample:**

- Bootstrap CIs provide better coverage for n < 100
- Cross-fitting (k-fold) reduces overfitting bias
- Weight clipping improves stability

See Also
--------

- :doc:`/theory/mathematical_foundations` - Detailed theoretical analysis
- :doc:`/guides/user_guide` - Practical usage patterns
- :doc:`/guides/troubleshooting` - Common issues and solutions