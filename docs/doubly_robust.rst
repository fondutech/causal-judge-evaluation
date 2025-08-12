Doubly Robust Estimation
========================

Doubly Robust (DR) estimation combines outcome modeling with importance sampling for better bias-variance tradeoffs.

Overview
--------

DR estimation uses the formula:

.. math::

   V_{DR}(\pi') = \mathbb{E}[g(X, A', S')] + \mathbb{E}[W \cdot (R - g(X, A, S))]

Where:
- First term: Direct method using outcome model
- Second term: IPS correction for residual bias
- g: Outcome model (cross-fitted)
- W: Importance weights (calibrated)

Basic Usage
-----------

.. code-block:: python

   from cje import DRCPOEstimator
   from cje.utils.fresh_draws import load_fresh_draws_from_jsonl
   
   # Create DR estimator
   dr = DRCPOEstimator(sampler, n_folds=5)
   
   # Add fresh draws for each target policy
   for policy in sampler.target_policies:
       # Load pre-generated samples from target policy
       fresh_draws = load_fresh_draws_from_jsonl(f"{policy}_fresh.jsonl")
       dr.add_fresh_draws(policy, fresh_draws)
   
   # Run estimation
   results = dr.fit_and_estimate()

Fresh Draws
-----------

DR requires "fresh draws" - samples generated from the target policy:

**Real Fresh Draws** (Best)

Generate actual samples from your target policy:

.. code-block:: python

   # Generate responses from target policy
   prompts = dataset.get_prompts()
   fresh_responses = []
   
   for prompt in prompts:
       response = generate_from_target_policy(prompt)
       judge_score = evaluate_with_judge(prompt, response)
       fresh_responses.append({
           "prompt_id": prompt.id,
           "response": response,
           "judge_score": judge_score
       })
   
   # Load into DR
   fresh_dataset = FreshDrawDataset(
       target_policy="gpt4",
       draws_per_prompt=1,
       samples=fresh_responses
   )
   dr.add_fresh_draws("gpt4", fresh_dataset)

**Synthetic Fresh Draws** (For Testing)

Create synthetic draws with controlled correlation:

.. code-block:: python

   from cje.utils.fresh_draws import create_synthetic_fresh_draws
   
   fresh_draws = create_synthetic_fresh_draws(
       dataset,
       target_policy="improved",
       draws_per_prompt=10,
       score_correlation=0.9,  # Correlation with logged scores
       seed=42
   )
   dr.add_fresh_draws("improved", fresh_draws)

Cross-fitting
-------------

DR uses cross-fitting to prevent overfitting:

1. Data split into k folds (default k=5)
2. For each fold, train outcome model on other k-1 folds
3. Predict on held-out fold
4. Ensures orthogonality between components

Architecture
------------

The DR implementation uses inheritance and composition:

.. code-block:: text

   DREstimator (inherits from CalibratedIPS)
   ├── Reuses all weight machinery
   ├── Adds outcome modeling
   └── Composes outcome model (not inherited)
   
   BaseOutcomeModel (abstract)
   ├── Handles cross-fitting infrastructure
   └── Subclasses implement single-model logic

Custom Outcome Models
---------------------

Implement custom outcome models by extending BaseOutcomeModel:

.. code-block:: python

   from cje import BaseOutcomeModel
   import xgboost as xgb
   
   class XGBoostOutcomeModel(BaseOutcomeModel):
       def __init__(self, n_folds=5, **xgb_params):
           super().__init__(n_folds)
           self.xgb_params = xgb_params
       
       def _fit_single_model(self, prompts, responses, rewards, judge_scores):
           # Extract features
           features = self._extract_features(prompts, responses, judge_scores)
           
           # Train XGBoost
           model = xgb.XGBRegressor(**self.xgb_params)
           model.fit(features, rewards)
           return model
       
       def _predict_single_model(self, model, prompts, responses, judge_scores):
           features = self._extract_features(prompts, responses, judge_scores)
           return model.predict(features)
       
       def _extract_features(self, prompts, responses, judge_scores):
           # Create feature matrix
           import numpy as np
           features = np.column_stack([
               [len(p.split()) for p in prompts],  # Prompt length
               [len(r.split()) for r in responses],  # Response length
               judge_scores  # Judge scores
           ])
           return features
   
   # Use custom model
   dr = DRCPOEstimator(
       sampler,
       outcome_model=XGBoostOutcomeModel(
           n_folds=5,
           n_estimators=100,
           max_depth=3
       )
   )

When to Use DR
--------------

**Use DR when:**

- You can generate samples from target policy
- You need lowest possible variance
- You have small to medium datasets
- Robustness is important (doubly robust property)

**Don't use DR when:**

- Cannot generate target samples
- Have very large datasets (IPS sufficient)
- Need fastest possible estimation

Performance Comparison
----------------------

Typical variance reduction with DR:

.. list-table::
   :header-rows: 1
   
   * - Method
     - Relative SE
     - Notes
   * - RawIPS
     - 1.00
     - Baseline
   * - CalibratedIPS
     - 0.60-0.80
     - 20-40% reduction
   * - DR-CPO
     - 0.30-0.50
     - 50-70% reduction

Next Steps
----------

- See :doc:`custom_outcome_models` for more examples
- See :doc:`api/core` for full API reference