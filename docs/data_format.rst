Data Format Guide
=================

This guide explains the data format requirements for CJE.

Required Fields
---------------

Each sample in your JSONL file must have these fields:

.. code-block:: json

   {
     "prompt": "string",
     "response": "string", 
     "base_policy_logprob": -12.345,
     "target_policy_logprobs": {
       "policy1": -10.123,
       "policy2": -11.456
     }
   }

Field Descriptions
------------------

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

Optional Fields
---------------

**reward** (float, 0-1)
   Pre-calibrated reward if already computed.
   If not present, will be computed from judge scores.

**metadata** (dict)
   Additional fields are automatically collected here:
   
   - ``judge_score``: AI judge evaluation score
   - ``oracle_label``: Ground truth label for calibration
   - ``prompt_id``: Unique identifier for the prompt
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

Judge Scores and Calibration
-----------------------------

Judge scores can be raw or calibrated:

**Raw Judge Scores**

.. code-block:: json

   {
     "metadata": {
       "judge_score": 7.5  
     }
   }

**Calibrated Rewards**

.. code-block:: python

   from cje import calibrate_dataset
   
   # Calibrate judge scores to oracle labels
   calibrated_dataset, stats = calibrate_dataset(
       dataset,
       judge_field="judge_score",
       oracle_field="human_rating"
   )
   
   # Now samples have calibrated rewards
   # sample.reward = calibrated score

Example: Complete Sample
------------------------

.. code-block:: json

   {
     "prompt": "Explain quantum computing to a 5-year-old",
     "response": "Quantum computing is like having a magic box...",
     "base_policy_logprob": -245.67,
     "target_policy_logprobs": {
       "gpt4": -198.45,
       "gpt4_cot": -203.12,
       "claude": -201.89
     },
     "metadata": {
       "prompt_id": "qc_explain_001",
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