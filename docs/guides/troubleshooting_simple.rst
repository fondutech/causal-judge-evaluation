Troubleshooting Guide
=====================

Quick diagnosis and solutions for common CJE issues.

Quick Diagnosis
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Symptom
     - Likely Cause
     - Quick Fix
   * - Wide confidence intervals
     - High weight variance
     - Use DR-CPO, more data, or similar policies
   * - Estimators disagree
     - Poor calibration
     - Enable oracle validation
   * - API errors (403/429)
     - Rate limits or auth
     - Check API key, reduce batch size
   * - Slow performance
     - Large dataset/model
     - Use max_samples, smaller judge
   * - Memory errors
     - Loading full dataset
     - Enable streaming/chunking
   * - No teacher forcing
     - Wrong provider
     - Use Fireworks or Together

Common Issues and Solutions
---------------------------

1. High Variance / Wide CIs
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Confidence intervals span most of [0,1] range

**Diagnosis**:

.. code-block:: python

   # Check effective sample size
   if results['diagnostics']['ess_percentage'] < 10:
       print("⚠️ Low ESS - high variance expected")

**Solutions**:

- Switch to DR estimator: ``estimator_name="DRCPO"``
- Reduce distribution shift: Use more similar policies
- Increase data: Need 10x more for high shift
- Clip weights: ``estimator.clip = 10.0``

2. API and Authentication
~~~~~~~~~~~~~~~~~~~~~~~~~

**403 Forbidden**:

.. code-block:: bash

   # Check API key is set
   echo $OPENAI_API_KEY
   echo $FIREWORKS_API_KEY
   
   # Set if missing
   export OPENAI_API_KEY="sk-..."

**429 Rate Limit**:

.. code-block:: yaml

   # Reduce batch size
   batch_size: 5  # Default is 10
   
   # Add delays
   api_delay: 1.0  # Seconds between calls

3. Teacher Forcing Issues
~~~~~~~~~~~~~~~~~~~~~~~~~

**"No teacher forcing support"**:

.. code-block:: python

   # Check provider compatibility
   print(f"Provider: {config.logging_policy.provider}")
   print(f"Supports TF: {provider in ['fireworks', 'together']}")
   
   # Solution: Use compatible provider
   config.logging_policy.provider = "fireworks"

4. Memory and Performance
~~~~~~~~~~~~~~~~~~~~~~~~~

**Out of Memory**:

.. code-block:: python

   # Limit dataset size
   config.dataset.max_samples = 1000
   
   # Process in chunks
   config.batch_size = 100
   config.enable_streaming = True

**Slow Evaluation**:

.. code-block:: python

   # Use faster estimator
   config.estimator.name = "SNIPS"  # Instead of MRDR
   
   # Simpler judge
   config.judge.model_name = "gpt-3.5-turbo"
   
   # Disable oracle
   config.oracle.enabled = False

5. Data Format Issues
~~~~~~~~~~~~~~~~~~~~~

**"Missing required field"**:

.. code-block:: python

   # Check data format
   required = ["prompt_id", "prompt", "response"]
   sample = dataset[0]
   missing = [f for f in required if f not in sample]
   print(f"Missing fields: {missing}")
   
   # Fix by adding fields
   for item in dataset:
       if "prompt_id" not in item:
           item["prompt_id"] = str(uuid.uuid4())

Debugging Tools
---------------

Enable Debug Mode
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Verbose logging
   config.debug = True
   config.log_level = "DEBUG"
   
   # Save intermediate results
   config.save_intermediates = True

Validation Before Running
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Always validate first
   try:
       config.validate()
   except ValidationError as e:
       print(f"Config error: {e}")

Check Diagnostics
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # After running
   diag = results['diagnostics']
   print(f"ESS: {diag['ess_percentage']:.1f}%")
   print(f"Max weight: {diag['max_weight_ratio']:.1f}x")
   print(f"Judge correlation: {diag.get('oracle_correlation', 'N/A')}")

Error Recovery
--------------

Checkpoint Recovery
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Enable checkpointing
   config.checkpoint_path = "./checkpoint.pkl"
   
   # Resume from checkpoint
   if os.path.exists(config.checkpoint_path):
       print("Resuming from checkpoint...")

Clean Up After Errors
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Remove corrupted files
   rm -f *.checkpoint.jsonl
   rm -f work_dir/cache/*
   
   # Clear Python cache
   find . -type d -name __pycache__ -exec rm -rf {} +

Quick Reference Card
--------------------

.. code-block:: text

   DIAGNOSIS FLOWCHART
   
   Wide CIs? → Check ESS → Use DR-CPO or more data
   API Error? → Check key → Reduce batch size
   Slow? → Limit samples → Use faster model
   Wrong format? → Check fields → Add missing
   Crashes? → Enable checkpoint → Check memory

When All Else Fails
-------------------

1. **Minimal Test**: Start with 10 samples
2. **Simple Config**: Use ``simple_config()`` defaults
3. **Check Examples**: Compare with ``configs/example_eval.yaml``
4. **GitHub Issues**: Search existing or create new

Emergency Config
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Minimal working config
   from cje.config.unified import simple_config
   
   config = simple_config(
       dataset_name="./test_10_samples.jsonl",
       logging_model="gpt-3.5-turbo",
       target_model="gpt-3.5-turbo",  # Same model = no shift
       judge_model="gpt-3.5-turbo",   # Cheap judge
       estimator_name="IPS"           # Simplest estimator
   )
   config.dataset.max_samples = 10
   config.debug = True
   
   # Should work if anything works
   results = config.run()

See Also
--------

- :doc:`comprehensive_usage` - Full configuration options
- :doc:`technical_implementation` - Understanding internals
- GitHub Issues - For unresolved problems