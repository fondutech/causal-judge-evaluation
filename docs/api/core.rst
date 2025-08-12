Core API Reference
==================

This section documents the core estimation classes.

Estimators
----------

.. automodule:: cje.core
   :members:
   :undoc-members:
   :show-inheritance:

CalibratedIPS
~~~~~~~~~~~~~

.. autoclass:: cje.core.CalibratedIPS
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

.. autoclass:: cje.core.RawIPS
   :members:
   :inherited-members:
   :show-inheritance:
   
   .. automethod:: __init__
   .. automethod:: fit
   .. automethod:: estimate

DRCPOEstimator
~~~~~~~~~~~~~~

.. autoclass:: cje.core.DRCPOEstimator
   :members:
   :inherited-members:
   :show-inheritance:
   
   .. automethod:: __init__
   .. automethod:: add_fresh_draws
   .. automethod:: fit
   .. automethod:: estimate

MRDREstimator
~~~~~~~~~~~~~

.. autoclass:: cje.core.mrdr.MRDREstimator
   :members:
   :inherited-members:
   :show-inheritance:
   
   .. automethod:: __init__
   .. automethod:: add_fresh_draws
   .. automethod:: fit
   .. automethod:: estimate

TMLEEstimator
~~~~~~~~~~~~~

.. autoclass:: cje.core.tmle.TMLEEstimator
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

.. autoclass:: cje.core.outcome_models.BaseOutcomeModel
   :members:
   :show-inheritance:
   
   .. automethod:: __init__
   .. automethod:: fit
   .. automethod:: predict
   .. automethod:: _fit_single_model
   .. automethod:: _predict_single_model

IsotonicOutcomeModel
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: cje.core.outcome_models.IsotonicOutcomeModel
   :members:
   :show-inheritance:

CalibratorBackedOutcomeModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: cje.core.outcome_models.CalibratorBackedOutcomeModel
   :members:
   :show-inheritance:

LinearOutcomeModel
~~~~~~~~~~~~~~~~~~

.. autoclass:: cje.core.outcome_models.LinearOutcomeModel
   :members:
   :show-inheritance: