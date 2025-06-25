Technical Implementation Guide
==============================

This guide explains the technical details of CJE's implementation: teacher forcing, log probability computation, weight processing, and the complete pipeline.

Overview of CJE Pipeline
------------------------

.. code-block:: text

   Dataset → Log Probs → Judge Scores → Weights → Estimation → Results
      ↓         ↓            ↓           ↓          ↓           ↓
   JSONL    π(a|x)      AI Scores    π'/π₀     DR-CPO    Rankings

Each step involves critical technical decisions that affect accuracy.

Teacher Forcing and Log Probabilities
-------------------------------------

The Challenge
~~~~~~~~~~~~~

CJE needs to compute ``π(response|context)`` - the probability a policy assigns to a specific response. Modern LLM APIs make this challenging:

**Chat Completions API** (what most providers offer):
   - Designed for generation, not probability computation
   - Returns log probs only for generated tokens
   - Includes special tokens and formatting

**Completions API** (what CJE needs):
   - Supports ``echo=True`` to get log probs for input text
   - No special token interference
   - Consistent probability computation

Critical Insight
~~~~~~~~~~~~~~~~

**Chat and completions APIs return different probabilities for the same text!** This breaks importance weighting if not handled correctly.

CJE's Two-Pass Solution
~~~~~~~~~~~~~~~~~~~~~~~

For providers supporting both APIs:

.. code-block:: python

   # Pass 1: Generate response using Chat API
   response = await client.chat.completions.create(
       model=model_name,
       messages=[
           {"role": "system", "content": system_prompt},
           {"role": "user", "content": context}
       ],
       temperature=temperature
   )
   generated_text = response.choices[0].message.content
   
   # Pass 2: Compute log probability using Completions API
   formatted_prompt = f"{system_prompt}\n\nUser: {context}\n\nAssistant: {generated_text}"
   
   logprob_response = await client.completions.create(
       model=model_name,
       prompt=formatted_prompt,
       max_tokens=0,  # Don't generate, just score
       echo=True,     # Return log probs for input
       logprobs=1
   )

Provider Compatibility
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Provider
     - Teacher Forcing
     - Method
     - Notes
   * - OpenAI
     - ❌ Removed
     - Generation only
     - Lost completions API in 2024
   * - Fireworks
     - ✅ Full
     - Completions API
     - Best for teacher forcing
   * - Together
     - ✅ Full
     - Completions API
     - Good alternative
   * - Anthropic
     - ❌ Never had
     - Generation only
     - Judge-only provider

Template Processing
~~~~~~~~~~~~~~~~~~~

Different APIs require different prompt formats:

.. code-block:: python

   # Chat format (for generation)
   messages = [
       {"role": "system", "content": "You are helpful."},
       {"role": "user", "content": "What is 2+2?"},
       {"role": "assistant", "content": "2+2 equals 4."}
   ]
   
   # Completions format (for log probs)
   prompt = """You are helpful.
   
   User: What is 2+2?
   
   Assistant: 2+2 equals 4."""
   
   # CJE handles this conversion automatically

Weight Processing Pipeline
--------------------------

Importance weights ``w = π'/π₀`` are central to off-policy evaluation. CJE implements sophisticated processing:

1. Raw Weight Computation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Basic importance weight
   log_ratio = target_logprob - logging_logprob
   weight = exp(log_ratio)
   
   # Numerical stability
   log_ratio = clip(log_ratio, -20, 20)  # Prevent overflow
   weight = exp(log_ratio)

2. Weight Diagnostics
~~~~~~~~~~~~~~~~~~~~~

CJE tracks critical metrics:

.. code-block:: python

   # Effective sample size
   weights_normalized = weights / sum(weights)
   ess = 1 / sum(weights_normalized ** 2)
   ess_percentage = (ess / n_samples) * 100
   
   # Weight statistics
   cv = std(weights) / mean(weights)  # Coefficient of variation
   max_weight = max(weights) / mean(weights)
   
   # Warnings
   if ess_percentage < 10:
       warn("Low ESS: High variance expected")
   if max_weight > 100:
       warn("Extreme weights detected")

3. Weight Stabilization
~~~~~~~~~~~~~~~~~~~~~~~

Multiple techniques to improve stability:

**Clipping**:

.. code-block:: python

   weights = clip(weights, 0, clip_value)

**Self-Normalization** (SNIPS):

.. code-block:: python

   normalized_weights = weights / sum(weights)

**Truncation**:

.. code-block:: python

   threshold = percentile(weights, 99)
   weights = minimum(weights, threshold)

4. Cross-Fitting
~~~~~~~~~~~~~~~~

Prevents overfitting in outcome models:

.. code-block:: python

   # Split data into k folds
   for fold in range(k):
       train_idx = [i for i in range(n) if i % k != fold]
       test_idx = [i for i in range(n) if i % k == fold]
       
       # Fit on train
       model.fit(X[train_idx], y[train_idx], weights[train_idx])
       
       # Predict on test
       predictions[test_idx] = model.predict(X[test_idx])

Judge Score Processing
----------------------

Judge scores undergo calibration and uncertainty quantification:

Score Calibration
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.isotonic import IsotonicRegression
   
   # Calibrate judge scores using oracle labels
   calibrator = IsotonicRegression(out_of_bounds='clip')
   calibrator.fit(proxy_scores, oracle_scores)
   
   # Apply to all scores
   calibrated_scores = calibrator.transform(all_proxy_scores)

Uncertainty Integration
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For confidence interval judges
   score = judge.score(context, response)
   # Returns: JudgeScoreWithCI(mean=0.8, variance=0.02, ci_lower=7, ci_upper=9)
   
   # Variance from CI width
   ci_width = score.ci_upper - score.ci_lower
   std_dev = ci_width / 3.92  # 95% CI = ±1.96σ
   variance = (std_dev / 10) ** 2  # Convert to 0-1 scale

Complete Pipeline Implementation
--------------------------------

Here's how all components work together:

.. code-block:: python

   class CJEPipeline:
       def run(self):
           # 1. Load and validate data
           dataset = self.load_dataset()
           self.validate_data(dataset)
           
           # 2. Compute log probabilities
           logging_logprobs = self.compute_logprobs(
               dataset, self.logging_policy, use_cache=True
           )
           
           target_logprobs = {}
           for policy in self.target_policies:
               target_logprobs[policy.name] = self.compute_logprobs(
                   dataset, policy, use_cache=True
               )
           
           # 3. Generate judge scores
           judge_scores = []
           for sample in dataset:
               score = self.judge.score(
                   sample['context'], 
                   sample['response']
               )
               judge_scores.append(score)
           
           # 4. Compute importance weights
           weights = {}
           for policy_name, logprobs in target_logprobs.items():
               log_ratios = logprobs - logging_logprobs
               log_ratios = np.clip(log_ratios, -20, 20)
               weights[policy_name] = np.exp(log_ratios)
           
           # 5. Run estimation
           estimator = self.create_estimator()
           results = {}
           
           for policy_name, policy_weights in weights.items():
               # Prepare estimation data
               rewards = [s.mean for s in judge_scores]
               variances = [s.variance for s in judge_scores]
               
               # Cross-fitted estimation
               estimate = estimator.estimate(
                   rewards=rewards,
                   weights=policy_weights,
                   reward_variances=variances,
                   cross_fit=True
               )
               
               results[policy_name] = estimate
           
           return results

Caching and Optimization
------------------------

Log Probability Caching
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # CJE automatically caches expensive computations
   cache_key = hash((model_name, prompt, temperature))
   
   if cache_key in cache:
       return cache[cache_key]
   
   result = compute_logprob(...)
   cache[cache_key] = result
   return result

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Process in batches for efficiency
   batch_processor = BatchProcessor(
       batch_size=10,
       checkpoint_manager=checkpoint_mgr
   )
   
   results = batch_processor.process_batches(
       items=dataset,
       process_fn=score_batch,
       description="Scoring responses"
   )

Parallel Execution
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Parallel API calls
   async def process_all(items):
       tasks = [process_item(item) for item in items]
       return await asyncio.gather(*tasks)

Numerical Stability
-------------------

Log-Space Computation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Work in log space to prevent overflow
   log_weights = target_logprobs - logging_logprobs
   log_weights = clip(log_weights, -20, 20)
   
   # Log-sum-exp trick for normalization
   max_log_weight = max(log_weights)
   log_normalizer = max_log_weight + log(sum(exp(log_weights - max_log_weight)))
   normalized_log_weights = log_weights - log_normalizer

Variance Computation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Stable variance computation
   def stable_variance(weights, rewards):
       n = len(weights)
       weighted_sum = sum(weights * rewards)
       weighted_sum_sq = sum(weights * rewards**2)
       
       # Bias-corrected variance
       mean = weighted_sum / sum(weights)
       var = (weighted_sum_sq / sum(weights)) - mean**2
       
       # Finite sample correction
       ess = sum(weights)**2 / sum(weights**2)
       var = var * n / (n - 1) * n / ess
       
       return var

Error Handling
--------------

Graceful Degradation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   try:
       # Try teacher forcing
       logprob = compute_teacher_forced_logprob(...)
   except APIError:
       # Fall back to approximation
       logprob = approximate_logprob_from_generation(...)
   except:
       # Last resort: uniform assumption
       logprob = -len(tokens) * log(vocab_size)

Diagnostic Reporting
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   diagnostics = {
       "ess_percentage": ess_pct,
       "max_weight_ratio": max_weight / mean_weight,
       "weight_cv": weight_cv,
       "n_extreme_weights": sum(weights > 10 * mean_weight),
       "judge_correlation": oracle_correlation,
       "calibration_rmse": calibration_error
   }
   
   if diagnostics["ess_percentage"] < 10:
       warnings.append("Low effective sample size")

Performance Considerations
--------------------------

**API Costs**
   - Cache all log probabilities
   - Batch API calls
   - Use cheaper models for initial tests

**Memory Usage**
   - Stream large datasets
   - Clear caches periodically
   - Use checkpointing for recovery

**Speed Optimization**
   - Parallel API calls
   - Vectorized operations
   - Efficient data structures

Best Practices
--------------

1. **Always validate teacher forcing compatibility** before running experiments
2. **Monitor weight diagnostics** - ESS < 10% indicates problems
3. **Use cross-fitting** for DR estimators to prevent overfitting
4. **Cache aggressively** - log probs are expensive
5. **Work in log space** for numerical stability
6. **Enable checkpointing** for long runs
7. **Validate calibration** with oracle samples

See Also
--------

- :doc:`comprehensive_usage` - User-focused guide
- :doc:`evaluation_methods` - High-level evaluation approaches
- :doc:`custom_components` - Extending CJE