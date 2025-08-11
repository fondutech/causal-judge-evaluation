Data API Reference
==================

This section documents the data models and loading utilities.

Data Models
-----------

Sample
~~~~~~

.. autoclass:: cje.data.Sample
   :members:
   :show-inheritance:

Dataset
~~~~~~~

.. autoclass:: cje.data.Dataset
   :members:
   :show-inheritance:

EstimationResult
~~~~~~~~~~~~~~~~

.. autoclass:: cje.data.EstimationResult
   :members:
   :show-inheritance:
   
   .. automethod:: best_policy
   .. automethod:: confidence_interval

PrecomputedSampler
~~~~~~~~~~~~~~~~~~

.. autoclass:: cje.data.PrecomputedSampler
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

.. autoclass:: cje.data.FreshDrawSample
   :members:
   :show-inheritance:

FreshDrawDataset
~~~~~~~~~~~~~~~~

.. autoclass:: cje.data.FreshDrawDataset
   :members:
   :show-inheritance:
   
   .. automethod:: get_scores_for_prompt_id
   .. automethod:: get_draws_for_prompt_id

Data Loading
------------

.. autofunction:: cje.load_dataset_from_jsonl

.. autofunction:: cje.data.add_rewards_to_existing_data

Factory Pattern
---------------

DatasetFactory
~~~~~~~~~~~~~~

.. autoclass:: cje.data.DatasetFactory
   :members:
   :show-inheritance:
   
   .. automethod:: create_from_jsonl
   .. automethod:: create_from_dict

DatasetLoader
~~~~~~~~~~~~~

.. autoclass:: cje.data.DatasetLoader
   :members:
   :show-inheritance: