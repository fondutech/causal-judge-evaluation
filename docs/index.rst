Causal Judge Evaluation Documentation
======================================

.. image:: img/CJE_logo.svg
   :align: center
   :alt: CJE Logo
   :width: 400px

Production-ready framework for unbiased LLM evaluation using causal inference.

Quick Start
-----------

Install and run your first evaluation:

.. code-block:: bash

   # Install
   pip install -e .
   
   # Set API key for log probability computation
   export FIREWORKS_API_KEY="your-key"

.. code-block:: python

   from cje import (
       load_dataset_from_jsonl,
       PrecomputedSampler, 
       CalibratedIPS
   )
   
   # Load data with precomputed log probabilities
   dataset = load_dataset_from_jsonl("data.jsonl")
   
   # Create sampler and estimator
   sampler = PrecomputedSampler(dataset)
   estimator = CalibratedIPS(sampler)
   
   # Get unbiased policy estimates
   results = estimator.fit_and_estimate()
   print(f"Best policy: {results.best_policy()}")

Key Features
------------

- **Unbiased Estimation**: Corrects for distribution shift between policies
- **Variance Control**: Isotonic calibration prevents weight explosion
- **Doubly Robust**: Optional DR estimation for better bias-variance tradeoff
- **Production Ready**: Clean API, comprehensive tests, type hints throughout

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   
   getting_started
   data_format
   estimators
   
.. toctree::
   :maxdepth: 2
   :caption: API Reference
   
   api/core
   api/data
   api/calibration
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Advanced
   
   doubly_robust
   custom_outcome_models

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`