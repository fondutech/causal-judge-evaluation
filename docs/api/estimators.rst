Estimators
==========

CJE-Core provides several off-policy evaluation estimators, each with different bias-variance trade-offs.

Overview
--------

All estimators inherit from the base :class:`~cje.estimators.base.Estimator` class and return 
:class:`~cje.estimators.results.EstimationResult` objects with standardized interfaces.

.. autoclass:: cje.estimators.base.Estimator
   :members:

.. autoclass:: cje.estimators.results.EstimationResult
   :members:

Inverse Propensity Scoring (IPS)
--------------------------------

The simplest importance sampling estimator.

**Mathematical Foundation:**

.. math::

   \hat{V}^{\text{IPS}}(\pi) = \frac{1}{n} \sum_{i=1}^n w_i r_i

where :math:`w_i = \frac{\pi(a_i|x_i)}{\pi_0(a_i|x_i)}` are importance weights.

**Characteristics:**
- ✅ Unbiased under correct propensity estimates
- ❌ High variance with distribution shift
- ⚡ Very fast computation

.. autoclass:: cje.estimators.ips.MultiIPSEstimator
   :members:
   :show-inheritance:

Self-Normalized IPS (SNIPS)
---------------------------

Reduces variance by normalizing importance weights.

**Mathematical Foundation:**

.. math::

   \hat{V}^{\text{SNIPS}}(\pi) = \frac{\sum_{i=1}^n w_i r_i}{\sum_{i=1}^n w_i}

**Characteristics:**
- ✅ Lower variance than IPS
- ❌ Introduces small bias
- ⚡ Fast computation

.. autoclass:: cje.estimators.ips.MultiSNIPSEstimator
   :members:
   :show-inheritance:

Doubly-Robust CPO (DR-CPO)
--------------------------

Combines importance sampling with outcome modeling for robustness.

**Mathematical Foundation:**

.. math::

   \hat{V}^{\text{DR}}(\pi) = \frac{1}{n} \sum_{i=1}^n \left[ \mu_\pi(x_i) + w_i(r_i - \mu(x_i, a_i)) \right]

**Characteristics:**
- ✅ Robust to model misspecification
- ✅ Lower variance than IPS
- ✅ Recommended for most use cases
- ⚠️ Requires outcome model fitting
- ⚠️ **Requires target policy samples** (``samples_per_policy`` ≥ 1)

.. warning::
   DR-CPO requires target policy samples to compute the baseline term :math:`\mu_\pi(x)`.
   Without them (``samples_per_policy=0``), DR reduces to IPW with no variance benefit.
   See :doc:`/estimators/dr_requirements` for details.

.. autoclass:: cje.estimators.drcpo.MultiDRCPOEstimator
   :members:
   :show-inheritance:

Model-Regularized DR (MRDR)
---------------------------

Advanced doubly-robust method with variance-optimized outcome models.

**Mathematical Foundation:**

.. math::

   \hat{V}^{\text{MRDR}}(\pi) = \frac{1}{n} \sum_{i=1}^n \left[ \mu_\pi^*(x_i) + w_i(r_i - \mu^*(x_i, a_i)) \right]

where :math:`\mu^*` minimizes the asymptotic variance.

**Characteristics:**
- ✅ Lowest variance estimator
- ✅ Optimal for small samples
- ✅ Handles large distribution shift well
- ⚠️ Computationally intensive
- ⚠️ **Requires target policy samples** (``samples_per_policy`` ≥ 1)

.. warning::
   Like DR-CPO, MRDR requires target policy samples. See :doc:`/estimators/dr_requirements`.

.. autoclass:: cje.estimators.mrdr.MultiMRDREstimator
   :members:
   :show-inheritance:

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

EstimationResult Methods
~~~~~~~~~~~~~~~~~~~~~~~~

The EstimationResult class provides several useful methods for analyzing results:

.. code-block:: python

   # Get confidence intervals
   ci_lower, ci_upper = result.confidence_interval(level=0.95)
   print(f"95% CI: [{ci_lower[0]:.3f}, {ci_upper[0]:.3f}]")
   
   # Rank policies from best to worst
   rankings = result.rank_policies()
   print(f"Policy rankings: {rankings}")
   
   # Compare two specific policies
   comparison = result.compare_policies(0, 1)
   print(f"Policy 0 vs 1: p-value = {comparison['p_value']:.3f}")
   
   # Get a human-readable summary
   print(result.summary())
   
   # Convert to dictionary for serialization
   result_dict = result.to_dict()

Bootstrap Confidence Intervals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For small samples (n < 100), bootstrap confidence intervals are automatically enabled:

.. code-block:: python

   # Bootstrap CIs are computed automatically for small samples
   result = estimator.estimate()
   
   if result.n < 100:
       # Bootstrap CIs available in metadata
       bootstrap_cis = result.metadata.get('bootstrap_confidence_intervals')
       if bootstrap_cis:
           print(f"Bootstrap 95% CI: {bootstrap_cis['95%']}")

Selection Guidelines
~~~~~~~~~~~~~~~~~~~~

Choose your estimator based on your data characteristics:

**IPS**: Use when you have:
- Large sample sizes (n > 1000)
- Small distribution shift
- Need fastest computation

**SNIPS**: Use when you have:
- Medium sample sizes (100 < n < 1000)
- Moderate distribution shift
- Want variance reduction over IPS

**DR-CPO**: Use when you have:
- Any sample size
- Moderate to large distribution shift
- Want robustness to model misspecification
- **Recommended for most use cases**

**MRDR**: Use when you have:
- Small sample sizes (n < 100)
- Large distribution shift
- Need maximum variance reduction
- Can afford computational cost

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**High Variance Estimates**

If your estimates have very large standard errors:

.. code-block:: python

   # Check effective sample size
   if result.metadata.get('ess_percentage', 100) < 10:
       print("⚠️  Low effective sample size - consider:")
       print("  • Using DR-CPO or MRDR instead of IPS/SNIPS")
       print("  • Collecting more data")
       print("  • Reducing distribution shift")

**Failed Convergence**
~~~~~~~~~~~~~~~~~~

If DR-CPO or MRDR fail to converge:

.. code-block:: python

   # Try simpler outcome model
   estimator = get_estimator("DRCPO", 
                           outcome_model_cls=Ridge,
                           outcome_model_kwargs={'alpha': 1.0})
   
   # Or fall back to IPS/SNIPS
   estimator = get_estimator("SNIPS", clip=10.0) 