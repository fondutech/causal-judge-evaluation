Doubly Robust Estimator Requirements
====================================

Overview
--------

Doubly Robust (DR) estimators in CJE (DR-CPO and MRDR) provide variance reduction compared to standard IPW by incorporating an outcome model. However, they have a critical requirement: **target policy samples**.

Why Target Samples Are Required
--------------------------------

Both DR-CPO and MRDR rely on the identity:

.. math::

   \psi_i = \mathbb{E}_{S \sim \pi'}[m(X_i,S)] + W_i(r_i - m(X_i,S_i))

Where:

- :math:`m(x,s)` is the outcome model (predicts reward given context and response)
- :math:`\mathbb{E}_{S \sim \pi'}[m(X_i,S)]` is the **baseline term** - expected outcome under the target policy
- :math:`W_i` is the importance weight
- :math:`r_i` is the observed reward

The baseline term is what provides the variance reduction. Without it:

- The formula reduces to :math:`\psi_i = 0 + W_i(r_i - m(X_i,S_i)) = W_i r_i - W_i m(X_i,S_i)`
- This is essentially IPW with an unhelpful bias term
- No variance reduction is achieved

Implementation in CJE
---------------------

Setting samples_per_policy
~~~~~~~~~~~~~~~~~~~~~~~~~~

When creating a DR estimator, the ``samples_per_policy`` parameter controls target sample generation:

.. code-block:: python

   # Proper DR with variance reduction
   estimator = MultiDRCPOEstimator(
       sampler=sampler,
       samples_per_policy=2,  # Generate 2 samples per policy (recommended)
       ...
   )

   # Degenerate DR (reduces to IPW)
   estimator = MultiDRCPOEstimator(
       sampler=sampler,
       samples_per_policy=0,  # ⚠️ No variance reduction!
       ...
   )

What Happens During Fitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. For each context in the dataset:

   - Generate ``samples_per_policy`` completions from each target policy
   - (Optionally) score them with the judge
   - Use them to estimate :math:`\mathbb{E}_{S \sim \pi'}[m(X_i,S)]`

2. The generated samples are automatically logged to ``work_dir/target_samples.jsonl``

3. If ``samples_per_policy=0``, a warning is displayed explaining the issue

Recommendations
---------------

For Maximum Benefit
~~~~~~~~~~~~~~~~~~~

- Set ``samples_per_policy=2`` (default) for good variance reduction
- Provide a ``judge_runner`` to score the generated samples
- Set ``score_target_policy_sampled_completions=True`` (default)

For Quick Testing
~~~~~~~~~~~~~~~~~

- Use IPW or SNIPW estimators instead of DR
- These don't require target samples and are still unbiased

For Production
~~~~~~~~~~~~~~

- Always use ``samples_per_policy ≥ 1`` with DR estimators
- Consider increasing to 5-10 for critical applications
- Monitor the logged target samples for quality

Common Pitfalls
---------------

1. **Setting samples_per_policy=0 for speed**: This completely negates the benefit of DR. Use IPW instead.

2. **Not providing a judge**: Target samples won't be scored, reducing outcome model accuracy.

3. **Using DR without understanding the requirement**: Always ensure your infrastructure can generate target samples.

Example Warning
---------------

When ``samples_per_policy=0``, you'll see:

.. code-block:: text

   ⚠️  WARNING: DR-CPO with samples_per_policy=0

   The DR-CPO estimator requires target policy samples to compute:
     μ_π(x) = E_π[μ(x,s)] (the baseline term)

   Without target samples (samples_per_policy=0):
     • The baseline term μ_π(x) = 0
     • DR reduces to standard IPW (no variance reduction)
     • Estimates represent differences from outcome model, not absolute values

   Recommended actions:
     1. Set samples_per_policy ≥ 1 for proper DR-CPO
     2. Use 'ipw' or 'snipw' estimators if target sampling is not possible