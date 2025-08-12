Data Format Guide
=================

This guide explains the data format requirements for CJE.

Required Fields
---------------

Core fields required in every sample:

.. code-block:: json

   {
     "prompt_id": "unique_id_123",
     "prompt": "string",
     "response": "string", 
     "base_policy_logprob": -12.345,
     "target_policy_logprobs": {
       "policy1": -10.123,
       "policy2": -11.456
     }
   }

For evaluation, you need one of these approaches:

**Option A: Pre-calibrated rewards**
   Include a ``reward`` field (0-1 range) in each sample

**Option B: Judge scores with oracle calibration** (recommended)
   - ``judge_score`` in metadata for ALL samples
   - ``oracle_label`` in metadata for calibration:
     - Absolute minimum: 10 samples (will error below this)
     - Recommended minimum: 50-100 samples for robust calibration
     - Best practice: 100+ samples for production use

Field Descriptions
------------------

**prompt_id** (string)
   Unique identifier for the prompt. Required for DR estimation, cross-validation, and analysis.

**prompt** (string)
   The input prompt or question

**response** (string)
   The generated response from your base policy

**base_policy_logprob** (float)
   Log probability of the response under your current/base policy.
   Use ``null`` if computation failed.

**target_policy_logprobs** (dict)
   Dictionary mapping target policy names to their log probabilities.
   Use ``null`` for failed computations.

Evaluation Fields
-----------------

**reward** (float, 0-1)
   Calibrated reward value. Either provided directly or computed from judge scores.

**metadata.judge_score** (float)
   AI judge evaluation score. Required if ``reward`` not provided.
   
**metadata.oracle_label** (float)  
   Ground truth label for calibration.
   - Minimum: 10 samples (will error if fewer)
   - Recommended: 50-100 samples for robust calibration
   - Without oracle labels, calibration is not possible

Optional Metadata
-----------------

Any additional fields in the data are automatically stored in ``metadata``:

- ``response_length``: Length of generated response
- ``generation_time``: Time to generate response
- ``cv_fold``: Pre-assigned cross-validation fold (if using cross-fitting)
- Any other custom fields

Handling Missing Data
---------------------

CJE automatically filters samples with missing log probabilities:

.. code-block:: python

   # Samples with null values are filtered
   {
     "base_policy_logprob": null,  # This sample will be excluded
     "target_policy_logprobs": {"gpt4": -32.1}
   }

Check how many samples remain after filtering:

.. code-block:: python

   sampler = PrecomputedSampler(dataset)
   print(f"Total samples: {sampler.n_samples}")
   print(f"Valid samples: {sampler.n_valid_samples}")
   
   if sampler.n_valid_samples < sampler.n_samples * 0.5:
       print("Warning: >50% of samples filtered!")

Judge Calibration Example
-------------------------

Calibrating judge scores improves accuracy:

.. code-block:: python

   from cje import load_dataset_from_jsonl, calibrate_dataset
   
   # Load data with judge scores and partial oracle labels
   dataset = load_dataset_from_jsonl("data.jsonl")
   
   # Calibrate judge scores to oracle labels
   calibrated_dataset, stats = calibrate_dataset(
       dataset,
       judge_field="judge_score",    # Field with judge scores (all samples)
       oracle_field="oracle_label"   # Field with oracle labels
   )
   
   print(f"Calibration used {stats.n_oracle} oracle samples")
   if stats.n_oracle < 50:
       print(f"⚠️  Warning: Only {stats.n_oracle} oracle samples. Consider 50-100 for robust calibration.")
   print(f"RMSE: {stats.calibration_rmse:.3f}")
   
   # Now all samples have calibrated rewards
   # sample.reward = calibrated score

Example: Complete Sample
------------------------

.. code-block:: json

   {
     "prompt_id": "qc_explain_001",
     "prompt": "Explain quantum computing to a 5-year-old",
     "response": "Quantum computing is like having a magic box...",
     "base_policy_logprob": -245.67,
     "target_policy_logprobs": {
       "gpt4": -198.45,
       "gpt4_cot": -203.12,
       "claude": -201.89
     },
     "metadata": {
       "judge_score": 8.5,
       "oracle_label": 0.85,
       "response_length": 127,
       "generation_time": 1.23
     }
   }

Creating Test Data
------------------

For testing, you can create synthetic data:

.. code-block:: python

   from cje import Sample, Dataset
   import json
   
   samples = []
   for i in range(100):
       sample = Sample(
           prompt_id=f"test_{i}",
           prompt=f"Question {i}",
           response=f"Answer {i}",
           base_policy_logprob=-10.0 - i*0.1,
           target_policy_logprobs={
               "improved": -9.0 - i*0.1
           },
           metadata={
               "judge_score": 0.5 + i*0.005
           }
       )
       samples.append(sample)
   
   dataset = Dataset(
       samples=samples,
       target_policies=["improved"]
   )
   
   # Save to JSONL
   with open("test_data.jsonl", "w") as f:
       for sample in samples:
           f.write(sample.model_dump_json() + "\n")

Best Practices
--------------

1. **Always validate log probabilities**: Ensure they're negative (log scale)
2. **Use consistent policy names**: Same names across all samples
3. **Include prompt IDs**: Helps with debugging and analysis
4. **Store failed computations as null**: Don't use magic numbers like -999
5. **Calibrate judge scores**: Improves estimate accuracy

Next Steps
----------

- See :doc:`getting_started` for basic usage
- See :doc:`api/data` for data model API reference