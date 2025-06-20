Troubleshooting Guide
=====================

*Comprehensive guide to diagnosing and fixing common CJE issues*

This guide consolidates all troubleshooting information from across the documentation. Find your issue below for quick solutions.

.. contents:: Quick Navigation
   :local:
   :depth: 2

🔧 Configuration Issues
-----------------------

Wide Confidence Intervals
~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**
- Very large standard errors
- Confidence intervals spanning most of [0, 1]
- Unreliable policy comparisons

**Causes & Solutions:**

1. **Policies too different from logging policy**
   
   - Check weight diagnostics for low ESS
   - Solution: Use more similar logging policy or collect more diverse data
   - Try SNIPS for automatic weight normalization

2. **Insufficient data**
   
   - Solution: Increase sample size (aim for 1000+ samples)
   - Use MRDR estimator which is more sample-efficient

3. **High variance in judge scores**
   
   - Check uncertainty diagnostics
   - Solution: Use better judge model or more oracle labels

Different Estimators Disagree
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**
- IPS and DRCPO give very different results
- MRDR estimate far from others

**Causes & Solutions:**

1. **Poor calibration**
   
   - Check calibration plot in outputs
   - Solution: Increase oracle labels for better calibration

2. **Model misspecification**
   
   - DRCPO/MRDR rely on outcome model
   - Solution: Try different feature sets or simpler models

3. **Extreme weights**
   
   - Check if IPS has very high variance
   - Solution: Use SNIPS or increase weight clipping

Configuration Validation Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Before running any experiment:**

.. code-block:: bash

   cje validate --cfg-path configs --cfg-name your_config

**Common errors:**

- Missing required fields (e.g., ``judge.template``)
- Invalid provider names
- Incompatible parameter combinations

🔑 API and Authentication
--------------------------

API Key Not Found
~~~~~~~~~~~~~~~~~

**Error:**

.. code-block:: text

   No provider API key found! Please set OPENAI_API_KEY or ANTHROPIC_API_KEY

**Solution:**

.. code-block:: bash

   # For OpenAI
   export OPENAI_API_KEY="sk-..."
   
   # For Anthropic
   export ANTHROPIC_API_KEY="sk-ant-..."
   
   # For Fireworks
   export FIREWORKS_API_KEY="..."

Rate Limiting
~~~~~~~~~~~~~

**Error:**

.. code-block:: text

   RateLimitError: You exceeded your current quota

**Solutions:**

1. **Reduce request volume:**
   
   - Decrease ``sample_limit`` in dataset config
   - Lower ``mc_samples`` for target policies
   - Use smaller ``oracle_fraction``

2. **Use cheaper models:**
   
   - Switch to ``gpt-3.5-turbo`` for logging/proxy
   - Use ``gpt-4o-mini`` instead of ``gpt-4o``

3. **Add delays:**
   
   - Implement request throttling
   - Use batch processing features

Module Import Errors
~~~~~~~~~~~~~~~~~~~~

**Error:**

.. code-block:: text

   ModuleNotFoundError: No module named 'cje'

**Solutions:**

.. code-block:: bash

   # Development installation
   cd causal-judge-evaluation
   pip install -e .
   
   # Or with Poetry
   poetry install

⚖️ Weight Processing Issues
---------------------------

Critical: ESS < 5%
~~~~~~~~~~~~~~~~~~

**This is the most serious weight issue - your results may be unreliable!**

**Diagnostic steps:**

1. **Check weight summary:**
   
   .. code-block:: text
   
      📊 Importance Weight Summary
      | Policy | ESS | Mean Weight | Status | Issues |
      | gpt-4  | 3.2% | 145.3 | ❌ CRITICAL | Extreme weights |

2. **Verify teacher forcing:**
   
   - Run with identical logging/target policy
   - Weights should be ≈ 1.0
   - If not, you have a probability computation bug

**Solutions by cause:**

- **Poor overlap** → Collect more diverse behavior data
- **Teacher forcing bugs** → Check API usage (chat vs completions)
- **Extreme prompts** → Use MRDR estimator or increase samples

Numerical Instability
~~~~~~~~~~~~~~~~~~~~~

**Symptoms:**
- NaN or Inf values in results
- Weights > 1e6 or < 1e-6
- Overflow warnings

**Solutions:**

1. **Check weight processing config:**
   
   .. code-block:: yaml
   
      diagnostics:
        log_ratio_clip: 20.0  # Reduce if needed
        weight_clip: 1000.0   # Hard maximum

2. **Use conservative mode:**
   
   .. code-block:: yaml
   
      weight_processing:
        mode: "conservative"
        hard_clip: [-10, 10]

Zero or Extreme Weights
~~~~~~~~~~~~~~~~~~~~~~~

**Diagnostic:**

.. code-block:: python

   # Check weight distribution
   print(f"Zero weights: {(weights == 0).sum()}")
   print(f"Max weight: {weights.max()}")
   print(f"Weight > 100: {(weights > 100).sum()}")

**Solutions:**
- Enable soft stabilization (default)
- Use SNIPS for automatic normalization
- Consider different logging policy

📊 Uncertainty and Calibration
------------------------------

High Gamma Values (γ > 3)
~~~~~~~~~~~~~~~~~~~~~~~~~

**Meaning:** Judge is overconfident (underestimates uncertainty)

**Solutions:**

1. **Switch judge approach:**
   
   - Try MC sampling with temperature > 0
   - Use explicit confidence prompting
   - Consider different judge model

2. **Adjust calibration:**
   
   - Increase oracle sample size
   - Use stratified sampling for oracle labels

Low ESS with Uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~

**When using variance shrinkage but still getting low ESS:**

1. **Adjust shrinkage:**
   
   .. code-block:: yaml
   
      uncertainty:
        variance_shrinkage:
          method: "adaptive"
          target_ess_fraction: 0.7  # Lower target

2. **Check variance distribution:**
   
   - High variance concentrated in few samples?
   - Consider fixed shrinkage with higher λ

Calibration Failures
~~~~~~~~~~~~~~~~~~~~

**Symptoms:**
- Isotonic calibration R² < 0.5
- Large gamma values
- Poor oracle-proxy correlation

**Solutions:**

1. **Increase oracle samples:**
   
   - Minimum 100 for reliable calibration
   - Use 20-30% oracle fraction

2. **Improve judge quality:**
   
   - Use better model for proxy judge
   - Simplify scoring rubric
   - Add clear examples to prompt

🚀 Performance Optimization
---------------------------

Slow Evaluation
~~~~~~~~~~~~~~~

**For large datasets or slow models:**

1. **Use faster estimators:**
   
   - IPS/SNIPS instead of DRCPO/MRDR
   - Reduce cross-validation folds

2. **Optimize API usage:**
   
   - Batch requests when possible
   - Use local models for development
   - Enable caching

3. **Reduce computational load:**
   
   .. code-block:: yaml
   
      target_policies:
        - mc_samples: 3  # Reduce from 5
          max_new_tokens: 512  # Reduce from 1024

Memory Issues
~~~~~~~~~~~~~

**For datasets > 10k samples:**

1. **Enable streaming mode** (if available)
2. **Process in batches**
3. **Monitor memory usage:**
   
   .. code-block:: bash
   
      # Linux
      htop
      
      # macOS
      # Use Activity Monitor

4. **Clear cache between runs:**
   
   .. code-block:: bash
   
      rm -rf outputs/*/cache/

High API Costs
~~~~~~~~~~~~~~

**Estimate costs before running:**

- Oracle calls: ``samples * oracle_fraction * $0.01``
- Proxy calls: ``samples * $0.0005``
- Target evaluation: ``n_policies * oracle_samples * $0.01``

**Cost reduction strategies:**

1. Use cheaper models for proxy/logging
2. Reduce oracle fraction to 15-20%
3. Decrease mc_samples for target policies
4. Use cached results when iterating

📋 Common Patterns
-----------------

Debugging Workflow
~~~~~~~~~~~~~~~~~~

When results seem wrong:

1. **Check basic statistics:**
   
   .. code-block:: python
   
      print(f"Dataset size: {len(data)}")
      print(f"Unique prompts: {data['prompt'].nunique()}")
      print(f"ESS: {results.get('ess_percentage', 'N/A')}%")

2. **Validate with simple baseline:**
   
   - Run with logging policy as target
   - Should get estimate ≈ mean(rewards)

3. **Compare multiple estimators:**
   
   - IPS vs DRCPO vs MRDR
   - Large differences indicate issues

4. **Check diagnostics:**
   
   - Weight distribution plots
   - Calibration curves
   - Guard rail violations

Best Practices Summary
~~~~~~~~~~~~~~~~~~~~~~

1. **Start small:** Test with 100 samples first
2. **Use defaults:** CJE defaults are well-tuned
3. **Monitor diagnostics:** Check ESS and calibration
4. **Compare methods:** Use multiple estimators
5. **Set seeds:** For reproducible debugging

Getting Help
~~~~~~~~~~~~

If these solutions don't resolve your issue:

1. Check the :doc:`/guides/user_guide` for your use case
2. Review :doc:`weight_processing` for technical details
3. See :doc:`uncertainty_evaluation` for uncertainty issues
4. Open an issue with:
   
   - Your configuration file
   - Error messages/logs
   - Dataset statistics
   - CJE version (`cje --version`)

.. seealso::

   - :doc:`weight_processing` - Technical details on weight computation
   - :doc:`uncertainty_evaluation` - Uncertainty-specific troubleshooting
   - :doc:`configuration_reference` - All configuration options