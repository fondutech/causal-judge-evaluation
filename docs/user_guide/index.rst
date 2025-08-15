User Guide
==========

Comprehensive guide for using CJE in production.

.. toctree::
   :maxdepth: 2
   
   installation
   data_preparation
   running_analysis
   interpreting_results

Overview
--------

This guide covers everything you need to know to use CJE effectively:

1. **Installation** - Setting up CJE in your environment
2. **Data Preparation** - Formatting your data correctly
3. **Running Analysis** - Choosing estimators and parameters
4. **Interpreting Results** - Understanding outputs and diagnostics

Getting Help
------------

- Check the :doc:`/modules/index` for detailed module documentation
- See :doc:`/theory/index` for theoretical foundations
- Look at :doc:`/examples/index` for working examples

Common Workflows
----------------

**Basic Off-Policy Evaluation**

.. code-block:: python

   from cje import analyze_dataset
   
   results = analyze_dataset(
       "logs.jsonl",
       estimator="calibrated-ips"
   )

**With Fresh Draws for DR**

.. code-block:: python

   results = analyze_dataset(
       "logs.jsonl",
       estimator="dr-cpo",
       fresh_draws_path="fresh_draws/"
   )

**Custom Oracle Coverage**

.. code-block:: python

   results = analyze_dataset(
       "logs.jsonl",
       estimator="calibrated-ips",
       oracle_coverage=0.2  # Use 20% for calibration
   )