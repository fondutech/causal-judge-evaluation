Utilities API Reference
=======================

This section documents utility functions and diagnostics.

Weight Diagnostics
------------------

.. autofunction:: cje.diagnose_weights

.. autofunction:: cje.create_weight_summary_table

.. autoclass:: cje.utils.WeightDiagnostics
   :members:
   :show-inheritance:
   
   .. automethod:: summary

Extreme Weights Analysis
------------------------

.. autofunction:: cje.analyze_extreme_weights

This function analyzes samples with extreme weights to identify patterns:

.. code-block:: python

   from cje import analyze_extreme_weights
   
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

.. autofunction:: cje.create_synthetic_fresh_draws

.. autofunction:: cje.load_fresh_draws_from_jsonl

.. autofunction:: cje.save_fresh_draws_to_jsonl

.. autofunction:: cje.validate_fresh_draws

Visualization (Optional)
------------------------

These functions are available if matplotlib is installed:

.. autofunction:: cje.plot_weight_dashboard

.. autofunction:: cje.plot_calibration_comparison

.. autofunction:: cje.plot_policy_estimates

Example usage:

.. code-block:: python

   from cje import plot_weight_dashboard
   
   # Create weight diagnostics dashboard
   fig, metrics = plot_weight_dashboard(
       raw_weights_dict,
       calibrated_weights_dict,
       n_samples=1000,
       save_path="weights.png"
   )

Teacher Forcing
---------------

.. autofunction:: cje.compute_teacher_forced_logprob

.. autofunction:: cje.compute_chat_logprob

.. autofunction:: cje.convert_chat_to_completions

Template Configurations
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: cje.teacher_forcing.ChatTemplateConfig
   :members:
   :show-inheritance:

.. autoclass:: cje.teacher_forcing.Llama3TemplateConfig
   :members:
   :show-inheritance:

.. autoclass:: cje.teacher_forcing.HuggingFaceTemplateConfig
   :members:
   :show-inheritance: