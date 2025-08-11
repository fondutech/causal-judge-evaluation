Calibration API Reference
=========================

This section documents the calibration utilities.

Judge Calibration
-----------------

.. autofunction:: cje.calibrate_dataset

.. autofunction:: cje.calibrate_judge_scores

JudgeCalibrator
~~~~~~~~~~~~~~~

.. autoclass:: cje.calibration.JudgeCalibrator
   :members:
   :show-inheritance:
   
   .. automethod:: __init__
   .. automethod:: fit
   .. automethod:: predict
   .. automethod:: fit_predict

CalibrationResult
~~~~~~~~~~~~~~~~~

.. autoclass:: cje.calibration.CalibrationResult
   :members:
   :show-inheritance:

Isotonic Calibration
--------------------

.. autofunction:: cje.calibrate_to_target_mean

Weight Calibration
------------------

The isotonic weight calibration is used internally by CalibratedIPS:

.. code-block:: python

   from cje import calibrate_to_target_mean
   
   # Calibrate weights to have mean 1.0
   calibrated_weights = calibrate_to_target_mean(
       raw_weights,
       target_mean=1.0,
       enforce_variance_nonincrease=True
   )

Parameters:

- **weights**: Raw importance weights
- **target_mean**: Target mean (usually 1.0)
- **enforce_variance_nonincrease**: Prevent variance explosion
- **max_variance_ratio**: Maximum allowed variance increase