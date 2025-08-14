Custom Outcome Models
=====================

This guide shows how to implement custom outcome models for DR estimation.

Base Class Overview
-------------------

All outcome models inherit from ``BaseOutcomeModel``:

.. code-block:: python

   from cje import BaseOutcomeModel
   
   class MyOutcomeModel(BaseOutcomeModel):
       def _fit_single_model(self, prompts, responses, rewards, judge_scores):
           """Train a single model (called k times for k-fold)"""
           pass
       
       def _predict_single_model(self, model, prompts, responses, judge_scores):
           """Make predictions with a single model"""
           pass

The base class handles:
- Cross-fitting infrastructure
- Fold assignment tracking
- Validation and error checking

You only implement single-model logic.

Example: Simple Linear Model
-----------------------------

A basic example using scikit-learn:

.. code-block:: python

   from sklearn.linear_model import Ridge
   from cje import BaseOutcomeModel
   import numpy as np
   
   class LinearOutcomeModel(BaseOutcomeModel):
       def __init__(self, n_folds=5, alpha=1.0):
           super().__init__(n_folds)
           self.alpha = alpha
       
       def _fit_single_model(self, prompts, responses, rewards, judge_scores):
           # Use judge scores as primary feature (isotonic baseline uses this alone)
           # Add simple text features
           features = np.column_stack([
               judge_scores,
               [len(r.split()) for r in responses],  # Response length
           ])
           
           model = Ridge(alpha=self.alpha)
           model.fit(features, rewards)
           return model
       
       def _predict_single_model(self, model, prompts, responses, judge_scores):
           features = np.column_stack([
               judge_scores,
               [len(r.split()) for r in responses],
           ])
           return model.predict(features)

Example: Embedding-Based Model
-------------------------------

Using sentence embeddings:

.. code-block:: python

   from sentence_transformers import SentenceTransformer
   from sklearn.ensemble import RandomForestRegressor
   from cje import BaseOutcomeModel
   import numpy as np
   
   class EmbeddingOutcomeModel(BaseOutcomeModel):
       def __init__(self, n_folds=5, model_name='all-MiniLM-L6-v2'):
           super().__init__(n_folds)
           self.encoder = SentenceTransformer(model_name)
       
       def _fit_single_model(self, prompts, responses, rewards, judge_scores):
           # Encode text to embeddings
           prompt_emb = self.encoder.encode(prompts)
           response_emb = self.encoder.encode(responses)
           
           # Combine features
           features = np.hstack([
               prompt_emb,
               response_emb,
               judge_scores.reshape(-1, 1)
           ])
           
           # Train forest
           model = RandomForestRegressor(n_estimators=100, max_depth=5)
           model.fit(features, rewards)
           return model
       
       def _predict_single_model(self, model, prompts, responses, judge_scores):
           # Same feature extraction
           prompt_emb = self.encoder.encode(prompts)
           response_emb = self.encoder.encode(responses)
           
           features = np.hstack([
               prompt_emb,
               response_emb,
               judge_scores.reshape(-1, 1)
           ])
           
           return model.predict(features)


Best Practices
--------------

1. **Start Simple**: The isotonic baseline is hard to beat
2. **Use Judge Scores**: They're your strongest signal
3. **Avoid Overfitting**: Cross-fitting handles this, but keep models simple
4. **Test Against Baseline**: Always compare to the default isotonic model

Testing Your Model
------------------

Test custom models before production use:

.. code-block:: python

   from cje import DRCPOEstimator, PrecomputedSampler
   from cje.utils.fresh_draws import create_synthetic_fresh_draws
   
   # Create test data
   test_dataset = create_test_dataset(n_samples=1000)
   sampler = PrecomputedSampler(test_dataset)
   
   # Test isotonic baseline
   dr_baseline = DRCPOEstimator(sampler)  # Uses IsotonicOutcomeModel
   
   # Test custom model
   dr_custom = DRCPOEstimator(
       sampler,
       outcome_model=MyCustomModel(n_folds=5)
   )
   
   # Add same fresh draws to both
   fresh = create_synthetic_fresh_draws(test_dataset, "target", draws_per_prompt=10)
   dr_baseline.add_fresh_draws("target", fresh)
   dr_custom.add_fresh_draws("target", fresh)
   
   # Compare results
   results_baseline = dr_baseline.fit_and_estimate()
   results_custom = dr_custom.fit_and_estimate()
   
   print(f"Baseline SE: {results_baseline.standard_errors[0]:.4f}")
   print(f"Custom SE: {results_custom.standard_errors[0]:.4f}")
   print(f"Variance reduction: {1 - results_custom.standard_errors[0]**2 / results_baseline.standard_errors[0]**2:.1%}")

Next Steps
----------

- See :doc:`doubly_robust` for DR overview
- See :doc:`api/core` for API reference