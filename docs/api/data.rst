Data API Reference
==================

This section documents the data models and loading utilities.

Data Models
-----------

Sample
~~~~~~

.. autoclass:: cje_simplified.data.Sample
   :members:
   :show-inheritance:

Dataset
~~~~~~~

.. autoclass:: cje_simplified.data.Dataset
   :members:
   :show-inheritance:

EstimationResult
~~~~~~~~~~~~~~~~

.. autoclass:: cje_simplified.data.EstimationResult
   :members:
   :show-inheritance:
   
   .. automethod:: best_policy
   .. automethod:: confidence_interval

PrecomputedSampler
~~~~~~~~~~~~~~~~~~

.. autoclass:: cje_simplified.data.PrecomputedSampler
   :members:
   :show-inheritance:
   
   .. automethod:: __init__
   .. automethod:: from_jsonl
   .. automethod:: compute_importance_weights
   .. automethod:: get_data_for_policy

Fresh Draws
-----------

FreshDrawSample
~~~~~~~~~~~~~~~

.. autoclass:: cje_simplified.data.FreshDrawSample
   :members:
   :show-inheritance:

FreshDrawDataset
~~~~~~~~~~~~~~~~

.. autoclass:: cje_simplified.data.FreshDrawDataset
   :members:
   :show-inheritance:
   
   .. automethod:: get_scores_for_prompt_id
   .. automethod:: get_draws_for_prompt_id

Data Loading
------------

.. autofunction:: cje_simplified.load_dataset_from_jsonl

.. autofunction:: cje_simplified.data.add_rewards_to_existing_data

Factory Pattern
---------------

DatasetFactory
~~~~~~~~~~~~~~

.. autoclass:: cje_simplified.data.DatasetFactory
   :members:
   :show-inheritance:
   
   .. automethod:: create_from_jsonl
   .. automethod:: create_from_dict

DatasetLoader
~~~~~~~~~~~~~

.. autoclass:: cje_simplified.data.DatasetLoader
   :members:
   :show-inheritance: