Core API Reference
==================

This section documents the core estimation classes.

Estimators
----------

.. automodule:: cje_simplified.core
   :members:
   :undoc-members:
   :show-inheritance:

CalibratedIPS
~~~~~~~~~~~~~

.. autoclass:: cje_simplified.core.CalibratedIPS
   :members:
   :inherited-members:
   :show-inheritance:
   
   .. automethod:: __init__
   .. automethod:: fit
   .. automethod:: estimate
   .. automethod:: fit_and_estimate
   .. automethod:: get_weights
   .. automethod:: get_diagnostics

RawIPS
~~~~~~

.. autoclass:: cje_simplified.core.RawIPS
   :members:
   :inherited-members:
   :show-inheritance:
   
   .. automethod:: __init__
   .. automethod:: fit
   .. automethod:: estimate

DRCPOEstimator
~~~~~~~~~~~~~~

.. autoclass:: cje_simplified.core.DRCPOEstimator
   :members:
   :inherited-members:
   :show-inheritance:
   
   .. automethod:: __init__
   .. automethod:: add_fresh_draws
   .. automethod:: fit
   .. automethod:: estimate

Outcome Models
--------------

BaseOutcomeModel
~~~~~~~~~~~~~~~~

.. autoclass:: cje_simplified.core.outcome_models.BaseOutcomeModel
   :members:
   :show-inheritance:
   
   .. automethod:: __init__
   .. automethod:: fit
   .. automethod:: predict
   .. automethod:: _fit_single_model
   .. automethod:: _predict_single_model

IsotonicOutcomeModel
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: cje_simplified.core.outcome_models.IsotonicOutcomeModel
   :members:
   :show-inheritance:

LinearOutcomeModel
~~~~~~~~~~~~~~~~~~

.. autoclass:: cje_simplified.core.outcome_models.LinearOutcomeModel
   :members:
   :show-inheritance: