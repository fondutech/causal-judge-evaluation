Estimators Guide
================

CJE provides several estimators with different bias-variance tradeoffs.

Quick Comparison
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 30 25 25

   * - Estimator
     - Best For
     - Pros
     - Cons
   * - **CalibratedIPS**
     - Most use cases
     - Variance control, stable
     - Slight bias
   * - **RawIPS**
     - Large datasets
     - Unbiased, simple
     - High variance
   * - **DRCPOEstimator**
     - When fresh draws available
     - Lowest variance
     - Requires target samples

CalibratedIPS (Recommended)
----------------------------

Uses isotonic calibration to control weight variance:

.. code-block:: python

   from cje import CalibratedIPS
   
   estimator = CalibratedIPS(
       sampler,
       enforce_variance_nonincrease=True,  # Prevent variance explosion (default)
       max_variance_ratio=1.0              # No variance increase allowed
   )
   results = estimator.fit_and_estimate()

**When to use:**

- Default choice for most applications
- When you have medium amounts of data (100-10K samples)
- When stability is more important than strict unbiasedness

**Key features:**

- Isotonic weight calibration
- Variance-safe blending
- Comprehensive diagnostics
- Automatic extreme weight handling

RawIPS
------

Standard importance sampling with optional weight clipping:

.. code-block:: python

   from cje import RawIPS
   
   estimator = RawIPS(
       sampler,
       clip_weight=100.0  # Clip weights at 100
   )
   results = estimator.fit_and_estimate()

**When to use:**

- Large datasets (>10K samples)  
- When unbiasedness is critical
- Quick prototyping

**Key features:**

- Unbiased estimation
- Simple and fast
- Weight clipping for stability
- No calibration overhead

DRCPOEstimator (Doubly Robust)
-------------------------------

Combines outcome modeling with IPS correction:

.. code-block:: python

   from cje import DRCPOEstimator, create_synthetic_fresh_draws
   
   # Create estimator with cross-fitted outcome model
   dr_estimator = DRCPOEstimator(sampler, n_folds=5)
   
   # Add fresh draws (samples from target policy)
   for policy in sampler.target_policies:
       fresh_draws = create_synthetic_fresh_draws(
           dataset, 
           target_policy=policy,
           draws_per_prompt=10
       )
       dr_estimator.add_fresh_draws(policy, fresh_draws)
   
   results = dr_estimator.fit_and_estimate()

**When to use:**

- Can generate samples from target policy
- Need lowest possible variance
- Small to medium datasets

**Key features:**

- Cross-fitted isotonic outcome model
- Doubly robust (consistent if either component correct)
- Requires fresh draws from target
- Best variance reduction

Understanding Weight Diagnostics
---------------------------------

All estimators provide weight diagnostics:

.. code-block:: python

   # Get diagnostics
   estimator = CalibratedIPS(sampler)
   results = estimator.fit_and_estimate()
   
   # Access diagnostics
   diagnostics = results.metadata['diagnostics']
   for policy in sampler.target_policies:
       diag = diagnostics[policy]
       print(f"{policy}:")
       print(f"  ESS: {diag['weights']['ess_fraction']:.1%}")
       print(f"  Max weight: {diag['weights']['max_weight']:.1f}")
       print(f"  Status: {diag['status']}")  # green/amber/red

**Key metrics:**

- **ESS (Effective Sample Size)**: Higher is better, >10% is good
- **Max weight**: Lower is better, <100 is good  
- **Tail ratio**: Weight concentration, <10 is good
- **Status**: Overall health (green/amber/red)

Choosing an Estimator
---------------------

**Start with CalibratedIPS** unless:

1. You have >10K samples → Consider RawIPS
2. You can generate target samples → Use DRCPOEstimator
3. You need strict unbiasedness → Use RawIPS with large clip_weight

**Decision flowchart:**

.. code-block:: text

   Can generate target samples?
   ├─ Yes → DRCPOEstimator
   └─ No → Have >10K samples?
           ├─ Yes → RawIPS
           └─ No → CalibratedIPS (default)

Custom Outcome Models (Advanced)
---------------------------------

For DR estimation, you can implement custom outcome models:

.. code-block:: python

   from cje import BaseOutcomeModel
   
   class MyOutcomeModel(BaseOutcomeModel):
       def _fit_single_model(self, prompts, responses, rewards, judge_scores):
           # Train your model
           model = train_model(prompts, responses, rewards)
           return model
       
       def _predict_single_model(self, model, prompts, responses, judge_scores):
           # Make predictions
           return model.predict(prompts, responses)
   
   # Use custom model
   dr_estimator = DRCPOEstimator(
       sampler,
       outcome_model=MyOutcomeModel(n_folds=5)
   )

The base class handles all cross-fitting complexity.

Next Steps
----------

- See :doc:`getting_started` for basic usage
- See :doc:`doubly_robust` for DR details
- See :doc:`api/core` for full API reference