Custom Components Guide
=======================

Extend CJE with custom estimators, judges, data loaders, and other components to fit your specific evaluation needs.

Overview
--------

CJE's modular architecture allows you to customize every aspect of the evaluation pipeline:

- **Custom Estimators**: Implement new causal inference methods
- **Custom Judges**: Create domain-specific evaluation criteria  
- **Custom Data Loaders**: Support new data formats
- **Custom Samplers**: Implement specialized sampling strategies
- **Custom Features**: Add domain-specific feature extraction

Component Architecture
----------------------

CJE uses a plugin-based architecture where components implement standard interfaces:

.. code-block:: text

   CJE Pipeline
   ├── Data Loader    → CJEDataset interface
   ├── Judge          → Judge interface  
   ├── Sampler        → MultiTargetSampler interface
   ├── Estimator      → Estimator interface
   └── Results        → EstimationResult interface

Each component can be swapped out independently, allowing modular customization.

Custom Estimators
-----------------

Creating a Custom Estimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement the ``Estimator`` interface:

.. code-block:: python

   from cje.estimators.base import Estimator
   from cje.estimators.results import EstimationResult
   import numpy as np

   class MyCustomEstimator(Estimator):
       def __init__(self, hyperparameter1=1.0, hyperparameter2="default"):
           super().__init__()
           self.hyperparameter1 = hyperparameter1
           self.hyperparameter2 = hyperparameter2
           
       def fit(self, dataset, sampler):
           """Fit the estimator to data"""
           self.dataset = dataset
           self.sampler = sampler
           
           # Your custom fitting logic here
           self._prepare_estimation()
           
       def estimate(self, policy_names):
           """Produce estimates for target policies"""
           estimates = []
           standard_errors = []
           
           for policy_name in policy_names:
               # Your custom estimation logic
               v_hat = self._compute_estimate(policy_name)
               se = self._compute_standard_error(policy_name)
               
               estimates.append(v_hat)
               standard_errors.append(se)
           
           return EstimationResult(
               v_hat=np.array(estimates),
               se=np.array(standard_errors),
               n=len(self.dataset),
               estimator_type=self.__class__.__name__,
               n_policies=len(policy_names),
               metadata={"hyperparameter1": self.hyperparameter1}
           )
           
       def _compute_estimate(self, policy_name):
           """Your custom estimation method"""
           # Example: simple importance sampling
           weights = self.sampler.importance_weights(policy_name)
           rewards = self.dataset.get_rewards()
           return np.mean(weights * rewards)
           
       def _compute_standard_error(self, policy_name):
           """Your custom uncertainty quantification"""
           # Example: bootstrap or analytical SE
           return 0.1  # Placeholder

Registration and Usage
~~~~~~~~~~~~~~~~~~~~~~

Register your estimator:

.. code-block:: python

   from cje.estimators import register_estimator

   # Register the estimator
   register_estimator("MyCustom", MyCustomEstimator)

   # Use in configuration
   estimator_config = {
       "name": "MyCustom",
       "hyperparameter1": 2.0,
       "hyperparameter2": "custom_value"
   }

Example: Regression Importance Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class RegressionIPS(Estimator):
       def __init__(self, model_type="linear", regularization=0.01):
           super().__init__()
           self.model_type = model_type
           self.regularization = regularization
           
       def fit(self, dataset, sampler):
           self.dataset = dataset
           self.sampler = sampler
           
           # Fit regression model for control variates
           from sklearn.linear_model import Ridge
           
           features = dataset.get_features()
           rewards = dataset.get_rewards()
           
           self.control_model = Ridge(alpha=self.regularization)
           self.control_model.fit(features, rewards)
           
       def estimate(self, policy_names):
           estimates = []
           for policy_name in policy_names:
               # Regression-adjusted importance sampling
               weights = self.sampler.importance_weights(policy_name)
               rewards = self.dataset.get_rewards()
               features = self.dataset.get_features()
               
               # Control variate adjustment
               predicted_rewards = self.control_model.predict(features)
               adjusted_rewards = rewards - predicted_rewards
               
               # Regression IPS estimate
               v_hat = (np.mean(predicted_rewards) + 
                       np.mean(weights * adjusted_rewards))
               
               estimates.append(v_hat)
           
           return EstimationResult(
               v_hat=np.array(estimates),
               se=self._bootstrap_se(policy_names),
               n=len(self.dataset),
               estimator_type="RegressionIPS"
           )

Custom Judges
-------------

Creating a Custom Judge
~~~~~~~~~~~~~~~~~~~~~~~

Implement the ``Judge`` interface:

.. code-block:: python

   from cje.judge.base import Judge
   import torch

   class DomainSpecificJudge(Judge):
       def __init__(self, domain="medical", criteria_weights=None):
           super().__init__()
           self.domain = domain
           self.criteria_weights = criteria_weights or {
               "accuracy": 0.4,
               "safety": 0.3, 
               "clarity": 0.3
           }
           
       def judge(self, contexts, responses):
           """Evaluate responses with domain-specific criteria"""
           scores = []
           
           for context, response in zip(contexts, responses):
               # Multi-criteria evaluation
               accuracy_score = self._evaluate_accuracy(context, response)
               safety_score = self._evaluate_safety(context, response)
               clarity_score = self._evaluate_clarity(context, response)
               
               # Weighted combination
               total_score = (
                   self.criteria_weights["accuracy"] * accuracy_score +
                   self.criteria_weights["safety"] * safety_score +
                   self.criteria_weights["clarity"] * clarity_score
               )
               
               scores.append(total_score)
               
           return np.array(scores)
           
       def _evaluate_accuracy(self, context, response):
           """Domain-specific accuracy evaluation"""
           if self.domain == "medical":
               return self._medical_accuracy_check(context, response)
           elif self.domain == "legal":
               return self._legal_accuracy_check(context, response) 
           else:
               return self._general_accuracy_check(context, response)

Model-Based Judges
~~~~~~~~~~~~~~~~~~

Use custom ML models for evaluation:

.. code-block:: python

   class BERTJudge(Judge):
       def __init__(self, model_path="bert-base-uncased", threshold=0.5):
           super().__init__()
           from transformers import AutoTokenizer, AutoModel
           
           self.tokenizer = AutoTokenizer.from_pretrained(model_path)
           self.model = AutoModel.from_pretrained(model_path)
           self.threshold = threshold
           
       def judge(self, contexts, responses):
           # Batch encode
           encodings = self.tokenizer(
               contexts, responses, 
               truncation=True, padding=True, return_tensors="pt"
           )
           
           # Get model predictions
           with torch.no_grad():
               outputs = self.model(**encodings)
               # Custom classification head
               scores = self._classify_quality(outputs.last_hidden_state)
               
           return scores.numpy()
           
       def _classify_quality(self, hidden_states):
           """Custom classification logic"""
           # Example: simple pooling + linear layer
           pooled = torch.mean(hidden_states, dim=1)
           scores = torch.sigmoid(self.classifier(pooled))
           return scores.squeeze()

Custom Data Loaders
-------------------

Creating Custom Dataset
~~~~~~~~~~~~~~~~~~~~~~~~

Implement the ``CJEDataset`` interface:

.. code-block:: python

   from cje.data.base import CJEDataset
   from cje.data.schema import CJESample

   class DatabaseDataset(CJEDataset):
       def __init__(self, connection_string, table_name, split="train"):
           super().__init__(name=f"db_{table_name}")
           self.connection_string = connection_string
           self.table_name = table_name
           self.split = split
           
       def __len__(self):
           # Query database for count
           query = f"SELECT COUNT(*) FROM {self.table_name} WHERE split='{self.split}'"
           return self._execute_query(query).fetchone()[0]
           
       def __getitem__(self, idx):
           # Load sample from database
           query = f"""
           SELECT uid, context, response, y_true, meta 
           FROM {self.table_name} 
           WHERE split='{self.split}' 
           LIMIT 1 OFFSET {idx}
           """
           row = self._execute_query(query).fetchone()
           
           return CJESample(
               uid=row['uid'],
               context=row['context'],
               response=row['response'],
               y_true=row['y_true'],
               meta=json.loads(row['meta']) if row['meta'] else {}
           )

Streaming Datasets
~~~~~~~~~~~~~~~~~~

For large datasets that don't fit in memory:

.. code-block:: python

   class StreamingDataset(CJEDataset):
       def __init__(self, data_stream, batch_size=1000):
           super().__init__(name="streaming")
           self.data_stream = data_stream
           self.batch_size = batch_size
           self._current_batch = []
           self._batch_idx = 0
           
       def __iter__(self):
           """Enable streaming iteration"""
           for batch in self.data_stream:
               self._current_batch = self._process_batch(batch)
               for sample in self._current_batch:
                   yield sample
                   
       def _process_batch(self, raw_batch):
           """Convert raw batch to CJESample objects"""
           samples = []
           for raw_sample in raw_batch:
               sample = CJESample(
                   uid=raw_sample['id'],
                   context=raw_sample['input'],
                   response=raw_sample['output'],
                   y_true=raw_sample.get('score'),
                   meta=raw_sample.get('metadata', {})
               )
               samples.append(sample)
           return samples

Custom Samplers
---------------

Specialized Sampling Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cje.loggers.base import MultiTargetSampler

   class TemperatureSweepSampler(MultiTargetSampler):
       def __init__(self, base_sampler, temperature_range=(0.1, 2.0), steps=10):
           super().__init__()
           self.base_sampler = base_sampler
           self.temperatures = np.linspace(*temperature_range, steps)
           
       def sample_responses(self, contexts, policy_configs):
           """Sample with temperature sweep for robustness"""
           all_responses = []
           all_logps = []
           
           for temp in self.temperatures:
               # Modify policy configs with current temperature
               temp_configs = self._add_temperature(policy_configs, temp)
               
               # Sample with current temperature
               responses, logps = self.base_sampler.sample_responses(
                   contexts, temp_configs
               )
               
               all_responses.extend(responses)
               all_logps.extend(logps)
               
           return all_responses, all_logps
           
       def _add_temperature(self, configs, temperature):
           """Add temperature to policy configurations"""
           temp_configs = []
           for config in configs:
               new_config = config.copy()
               new_config['temperature'] = temperature
               temp_configs.append(new_config)
           return temp_configs

Active Learning Samplers
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class ActiveLearningSampler(MultiTargetSampler):
       def __init__(self, base_sampler, uncertainty_threshold=0.5):
           super().__init__()
           self.base_sampler = base_sampler
           self.uncertainty_threshold = uncertainty_threshold
           
       def sample_responses(self, contexts, policy_configs):
           # Initial sampling
           responses, logps = self.base_sampler.sample_responses(
               contexts, policy_configs
           )
           
           # Identify high-uncertainty samples
           uncertainties = self._compute_uncertainty(responses, logps)
           uncertain_indices = np.where(uncertainties > self.uncertainty_threshold)[0]
           
           # Additional sampling for uncertain cases
           if len(uncertain_indices) > 0:
               uncertain_contexts = [contexts[i] for i in uncertain_indices]
               additional_responses, additional_logps = self.base_sampler.sample_responses(
                   uncertain_contexts, policy_configs
               )
               
               # Merge results
               responses, logps = self._merge_samples(
                   responses, logps, additional_responses, additional_logps, uncertain_indices
               )
               
           return responses, logps

Custom Features
---------------

Domain-Specific Feature Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from cje.features.base import FeatureExtractor

   class MedicalFeatureExtractor(FeatureExtractor):
       def __init__(self):
           super().__init__()
           # Load medical domain resources
           self.medical_terms = self._load_medical_dictionary()
           self.symptom_patterns = self._load_symptom_patterns()
           
       def extract_features(self, samples):
           """Extract medical domain features"""
           features = []
           
           for sample in samples:
               context_features = self._extract_context_features(sample.context)
               response_features = self._extract_response_features(sample.response)
               interaction_features = self._extract_interaction_features(
                   sample.context, sample.response
               )
               
               combined_features = {
                   **context_features,
                   **response_features,
                   **interaction_features
               }
               
               features.append(combined_features)
               
           return features
           
       def _extract_context_features(self, context):
           """Extract features from user context"""
           return {
               "symptom_count": self._count_symptoms(context),
               "urgency_level": self._assess_urgency(context),
               "medical_complexity": self._assess_complexity(context)
           }
           
       def _extract_response_features(self, response):
           """Extract features from assistant response"""
           return {
               "medical_accuracy": self._check_medical_accuracy(response),
               "safety_compliance": self._check_safety(response),
               "empathy_score": self._measure_empathy(response)
           }

Configuration Integration
-------------------------

Registering Custom Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Register all custom components
   from cje.registry import ComponentRegistry

   # Estimators
   ComponentRegistry.register_estimator("RegressionIPS", RegressionIPS)
   ComponentRegistry.register_estimator("MyCustom", MyCustomEstimator)

   # Judges  
   ComponentRegistry.register_judge("DomainSpecific", DomainSpecificJudge)
   ComponentRegistry.register_judge("BERT", BERTJudge)

   # Datasets
   ComponentRegistry.register_dataset("Database", DatabaseDataset)
   ComponentRegistry.register_dataset("Streaming", StreamingDataset)

YAML Configuration
~~~~~~~~~~~~~~~~~~

Use custom components in configuration:

.. code-block:: yaml

   # Dataset configuration (custom loader)
   dataset:
     name: "Database"
     connection_string: "postgresql://user:pass@localhost/db"
     table_name: "evaluations"
     split: "test"
   
   # Logging policy (what generated the historical data)
   logging_policy:
     provider: "openai"
     model_name: "gpt-3.5-turbo"
     temperature: 0.7
   
   # Target policies (what we want to evaluate)
   target_policies:
     - name: "test_policy"
       provider: "openai"
       model_name: "gpt-4o-mini"
       temperature: 0.7
       mc_samples: 5               # Monte Carlo samples per context
     
   # Judge configuration (custom judge)
   judge:
     name: "DomainSpecific" 
     domain: "medical"
     criteria_weights:
       accuracy: 0.5
       safety: 0.3
       clarity: 0.2
       
   # Estimator configuration (custom estimator)
   estimator:
     name: "RegressionIPS"
     model_type: "ridge"
     regularization: 0.05
     
   # Sampler configuration (custom sampler)
   sampler:
     name: "TemperatureSweep"
     temperature_range: [0.2, 1.5]
     steps: 8

Testing Custom Components
-------------------------

Unit Testing
~~~~~~~~~~~~

.. code-block:: python

   import unittest
   from cje.testing import create_mock_dataset

   class TestCustomEstimator(unittest.TestCase):
       def setUp(self):
           self.dataset = create_mock_dataset(n_samples=100)
           self.estimator = MyCustomEstimator()
           
       def test_estimation_accuracy(self):
           """Test estimator produces reasonable results"""
           self.estimator.fit(self.dataset, mock_sampler)
           results = self.estimator.estimate(["policy1", "policy2"])
           
           # Verify output format
           self.assertIsInstance(results, EstimationResult)
           self.assertEqual(len(results.v_hat), 2)
           
           # Verify estimates are reasonable
           self.assertTrue(all(0 <= v <= 1 for v in results.v_hat))

Integration Testing
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def test_end_to_end_pipeline():
       """Test custom components work together"""
       config = {
           "dataset": {"name": "Database", "table_name": "test_data"},
           "judge": {"name": "DomainSpecific", "domain": "medical"}, 
           "estimator": {"name": "RegressionIPS"}
       }
       
       # Run full pipeline
       results = run_cje_pipeline(config)
       
       # Verify results
       assert results.success
       assert len(results.estimates) > 0

Best Practices
--------------

**Component Design:**

- Follow single responsibility principle
- Implement comprehensive error handling
- Provide clear documentation and type hints
- Include configuration validation

**Performance:**

- Optimize for batch processing when possible
- Implement caching for expensive operations
- Use appropriate data structures for your use case
- Profile and benchmark custom components

**Testing:**

- Write comprehensive unit tests
- Test edge cases and error conditions
- Validate against known baselines
- Include integration tests with other components

**Documentation:**

- Document all parameters and their effects
- Provide usage examples
- Explain the theoretical foundation
- Include performance characteristics

This modular approach allows you to extend CJE for any evaluation scenario while maintaining compatibility with the existing ecosystem. 