Custom Outcome Models
=====================

This guide shows how to implement custom outcome models for DR estimation.

Base Class Overview
-------------------

All outcome models inherit from ``BaseOutcomeModel``:

.. code-block:: python

   from cje_simplified import BaseOutcomeModel
   
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

Example: Neural Network Model
------------------------------

Using a simple neural network:

.. code-block:: python

   import torch
   import torch.nn as nn
   from cje_simplified import BaseOutcomeModel
   
   class NeuralOutcomeModel(BaseOutcomeModel):
       def __init__(self, n_folds=5, hidden_dim=64, epochs=100):
           super().__init__(n_folds)
           self.hidden_dim = hidden_dim
           self.epochs = epochs
       
       def _fit_single_model(self, prompts, responses, rewards, judge_scores):
           # Extract features
           features = self._extract_features(prompts, responses, judge_scores)
           
           # Define simple network
           model = nn.Sequential(
               nn.Linear(features.shape[1], self.hidden_dim),
               nn.ReLU(),
               nn.Linear(self.hidden_dim, 1),
               nn.Sigmoid()
           )
           
           # Train
           optimizer = torch.optim.Adam(model.parameters())
           criterion = nn.MSELoss()
           
           X = torch.FloatTensor(features)
           y = torch.FloatTensor(rewards).reshape(-1, 1)
           
           for epoch in range(self.epochs):
               optimizer.zero_grad()
               outputs = model(X)
               loss = criterion(outputs, y)
               loss.backward()
               optimizer.step()
           
           return model
       
       def _predict_single_model(self, model, prompts, responses, judge_scores):
           features = self._extract_features(prompts, responses, judge_scores)
           X = torch.FloatTensor(features)
           
           with torch.no_grad():
               predictions = model(X).numpy().flatten()
           
           return predictions
       
       def _extract_features(self, prompts, responses, judge_scores):
           # Simple features - extend as needed
           import numpy as np
           return np.column_stack([
               [len(p.split()) for p in prompts],
               [len(r.split()) for r in responses],
               judge_scores
           ])

Example: Embedding-Based Model
-------------------------------

Using sentence embeddings:

.. code-block:: python

   from sentence_transformers import SentenceTransformer
   from sklearn.ensemble import RandomForestRegressor
   from cje_simplified import BaseOutcomeModel
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

Example: Prompt-Aware Model
----------------------------

Model that learns prompt-specific patterns:

.. code-block:: python

   from collections import defaultdict
   from cje_simplified import BaseOutcomeModel
   import numpy as np
   
   class PromptAwareOutcomeModel(BaseOutcomeModel):
       def __init__(self, n_folds=5, min_samples_per_prompt=5):
           super().__init__(n_folds)
           self.min_samples = min_samples_per_prompt
       
       def _fit_single_model(self, prompts, responses, rewards, judge_scores):
           # Group by prompt
           prompt_models = {}
           prompt_groups = defaultdict(list)
           
           for i, prompt in enumerate(prompts):
               prompt_groups[prompt].append(i)
           
           # Fit per-prompt models where enough data
           from sklearn.isotonic import IsotonicRegression
           
           for prompt, indices in prompt_groups.items():
               if len(indices) >= self.min_samples:
                   # Prompt-specific model
                   scores = judge_scores[indices]
                   rewards_subset = rewards[indices]
                   model = IsotonicRegression()
                   model.fit(scores, rewards_subset)
                   prompt_models[prompt] = model
           
           # Global fallback model
           global_model = IsotonicRegression()
           global_model.fit(judge_scores, rewards)
           
           return {
               'prompt_models': prompt_models,
               'global_model': global_model
           }
       
       def _predict_single_model(self, model, prompts, responses, judge_scores):
           predictions = np.zeros(len(prompts))
           
           for i, prompt in enumerate(prompts):
               if prompt in model['prompt_models']:
                   # Use prompt-specific model
                   predictions[i] = model['prompt_models'][prompt].predict(
                       [judge_scores[i]]
                   )[0]
               else:
                   # Use global model
                   predictions[i] = model['global_model'].predict(
                       [judge_scores[i]]
                   )[0]
           
           return predictions

Best Practices
--------------

1. **Feature Engineering**: Good features are crucial
   - Text length, complexity metrics
   - Embeddings for semantic understanding
   - Judge score transformations
   
2. **Regularization**: Prevent overfitting
   - Use L1/L2 regularization
   - Limit model complexity
   - Early stopping for neural networks
   
3. **Validation**: Check model quality
   - Monitor out-of-fold performance
   - Compare to isotonic baseline
   - Check for systematic biases

4. **Efficiency**: Keep models fast
   - Cache embeddings if reused
   - Use batch processing
   - Consider model size vs accuracy tradeoff

Testing Your Model
------------------

Test custom models before production use:

.. code-block:: python

   from cje_simplified import DRCPOEstimator, create_synthetic_fresh_draws
   
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