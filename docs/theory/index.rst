Mathematical Foundations
=======================

This section covers the theoretical foundations of Causal Judge Evaluation (CJE), including the mathematical formulations, assumptions, and theoretical guarantees of different estimators.

.. toctree::
   :maxdepth: 2
   :caption: Theory Topics

   mathematical_foundations
   estimator_comparison
   trajectory_methods

Overview
--------

CJE uses causal inference methods to estimate how policy changes would affect real-world outcomes using historical data. The key insight is that we can use importance sampling combined with outcome modeling to create unbiased estimates of policy value.

**Core Problem**: Given interaction logs collected under a logging policy π₀ and any target policy π' we might deploy, estimate the expected utility V(π') if all traffic were served by π'.

**Mathematical Foundation**: CJE implements doubly-robust estimators that are unbiased if either the importance weights or the outcome model is correctly specified.

Key Concepts
-----------

**Importance Sampling**
   Reweight historical data to simulate what would have happened under a different policy

**Doubly-Robust Estimation**
   Combine importance sampling with outcome modeling for robustness against model misspecification

**Cross-Fitting**
   Use sample splitting to avoid overfitting bias in nuisance function estimation

**Multiple Policy Evaluation**
   Jointly evaluate multiple policies with proper covariance estimation for statistical comparisons

Next Steps
----------

- :doc:`mathematical_foundations` - Complete mathematical formulation
- :doc:`estimator_comparison` - Detailed comparison of all estimators  
- :doc:`trajectory_methods` - Extensions to multi-step agent trajectories
- :doc:`../api/estimators` - Implementation details and usage examples 