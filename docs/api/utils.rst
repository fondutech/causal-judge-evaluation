Utilities API Reference
=======================

This section documents utility functions and diagnostics.

Weight Diagnostics
------------------

.. autofunction:: cje_simplified.diagnose_weights

.. autofunction:: cje_simplified.create_weight_summary_table

.. autoclass:: cje_simplified.utils.WeightDiagnostics
   :members:
   :show-inheritance:
   
   .. automethod:: summary

Extreme Weights Analysis
------------------------

.. autofunction:: cje_simplified.analyze_extreme_weights

This function analyzes samples with extreme weights to identify patterns:

.. code-block:: python

   from cje_simplified import analyze_extreme_weights
   
   json_report, text_report = analyze_extreme_weights(
       dataset=dataset,
       sampler=sampler,
       raw_weights_dict={"policy1": raw_weights},
       calibrated_weights_dict={"policy1": cal_weights},
       n_extreme=10,  # Analyze top 10 extreme samples
       output_dir="./analysis"
   )

Fresh Draw Utilities
--------------------

.. autofunction:: cje_simplified.create_synthetic_fresh_draws

.. autofunction:: cje_simplified.load_fresh_draws_from_jsonl

.. autofunction:: cje_simplified.save_fresh_draws_to_jsonl

.. autofunction:: cje_simplified.validate_fresh_draws

Visualization (Optional)
------------------------

These functions are available if matplotlib is installed:

.. autofunction:: cje_simplified.plot_weight_dashboard

.. autofunction:: cje_simplified.plot_calibration_comparison

.. autofunction:: cje_simplified.plot_policy_estimates

Example usage:

.. code-block:: python

   from cje_simplified import plot_weight_dashboard
   
   # Create weight diagnostics dashboard
   fig, metrics = plot_weight_dashboard(
       raw_weights_dict,
       calibrated_weights_dict,
       n_samples=1000,
       save_path="weights.png"
   )

Teacher Forcing
---------------

.. autofunction:: cje_simplified.compute_teacher_forced_logprob

.. autofunction:: cje_simplified.compute_chat_logprob

.. autofunction:: cje_simplified.convert_chat_to_completions

Template Configurations
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: cje_simplified.teacher_forcing.ChatTemplateConfig
   :members:
   :show-inheritance:

.. autoclass:: cje_simplified.teacher_forcing.Llama3TemplateConfig
   :members:
   :show-inheritance:

.. autoclass:: cje_simplified.teacher_forcing.HuggingFaceTemplateConfig
   :members:
   :show-inheritance: