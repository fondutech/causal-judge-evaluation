Choosing an Estimator
=====================

Defaults
--------

- Non‑DR (no fresh draws): ``calibrated-ips`` — IPS with SIMCal weight calibration
- DR (with fresh draws): ``stacked-dr`` — calibrated stacked doubly‑robust ensemble

Auto Selection
--------------

``--estimator auto`` (CLI) or ``estimator="auto"`` (API) selects:

- ``stacked-dr`` if ``--fresh-draws-dir`` / ``fresh_draws_dir`` is provided
- ``calibrated-ips`` otherwise

When to Use Which
-----------------

- Use ``calibrated-ips`` when you only have logged data and judge scores
  - SIMCal stabilizes weights, improves ESS, and adds honest CIs via oracle slice augmentation
- Use ``stacked-dr`` when you can collect fresh draws from target policies
  - DR improves efficiency and robustness; the stacked variant mixes DR‑CPO, TMLE, MRDR

Advanced Notes
--------------

- ``raw-ips`` is available for baselines and diagnostics
- Individual DR estimators (``dr-cpo``, ``tmle``, ``mrdr``) are available for ablations
- SIMCal and calibration support are used where applicable in both IPS and DR paths

See also
--------

- DR quickstart: :doc:`../tutorials/dr_quickstart`
- How it works: :doc:`../explanation/how_it_works`
