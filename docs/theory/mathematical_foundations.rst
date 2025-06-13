Mathematical Foundations
========================

This document covers the mathematical foundations and high-level architecture of CJE (code-agnostic).

.. important::
   **Terminology Mapping**: This implementation extends the baseline paper with additional robustness features. The table below maps paper concepts to code and configuration:

.. list-table:: Paper ↔ Implementation ↔ Configuration Mapping
   :header-rows: 1
   :widths: 25 25 25 25

   * - Paper Term
     - Implementation Class
     - Config Option
     - Description
   * - Calibrated DR-CPO
     - ``MultiDRCPOEstimator``
     - ``name: "DRCPO"``
     - Main estimator (Algorithm 1)
   * - Judge calibration
     - ``cross_fit_calibration``
     - ``judge:`` section
     - Score → reward mapping
   * - Weight calibration  
     - ``calibrate_weights_isotonic``
     - ``calibrate_weights: true``
     - Importance weight centering
   * - MRDR upgrade
     - ``MultiMRDREstimator``
     - ``name: "MRDR"``
     - Optional variance reduction
   * - Oracle slice
     - 25% holdout in cross-fitting
     - ``y_true`` field in data
     - Ground truth for calibration

**Implementation Enhancements** *(beyond paper baseline)*:
   * **Outcome calibration**: ``calibrate_outcome: true`` *(additional robustness)*
   * **Multi-policy**: Native :math:`K`-policy evaluation with covariance matrices
   * **Production hardening**: Automatic diagnostics, numerical stabilization

Problem Statement
-----------------

Given interaction logs collected under a stochastic **logging policy** :math:`\pi_0` and any **target policy** :math:`\pi'` we might deploy in the future, we want to estimate:

.. math::

   V(\pi') = \mathbb{E}_{X\sim P}\;\mathbb{E}_{S\sim \pi'(\cdot\mid X)}\bigl[\,Y(X,S)\bigr]

the **expected utility** (true KPI) if all traffic were served by :math:`\pi'`.

Causal Ingredients
------------------

.. list-table:: Key Components
   :header-rows: 1
   :widths: 20 40 40

   * - Notation
     - Meaning
     - Logged or learned?
   * - :math:`X`
     - Context (prompt, user state)
     - logged
   * - :math:`S`
     - Sequence or action produced
     - logged
   * - :math:`\pi_0(S\!\mid\!X)`
     - Exact probability of :math:`S` under the logger
     - logged (``logprobs = true``, temp ≥ 0.3)
   * - :math:`r`
     - **Calibrated reward** ≈ true utility
     - derived (judge → calibration)
   * - :math:`m(X,S)`
     - Outcome model predicting :math:`r`
     - learned on held-out folds

Assumptions
-----------

1. **Overlap**: :math:`\pi'(S\!\mid\!X)>0 \;\Rightarrow\; \pi_0(S\!\mid\!X)>0`
   
   *(Guaranteed when the logger is mildly stochastic.)*

2. **Ignorability**: :math:`Y(S)\perp\!\perp S \mid X` **in the logging data** (assumed)
   
   *After calibration we **treat** the calibrated reward* :math:`r` *as an unbiased stand-in for* :math:`Y`. *Ignorability still rests on conditioning on all confounders in* :math:`X` *(and, if needed, step variables* :math:`S`*); calibration alone cannot manufacture ignorability.*

3. **Consistency**: Observed reward equals the potential outcome for the served sequence.

Efficient Doubly-Robust Estimator (DR-CPO)
-------------------------------------------

Efficient Influence Function (EIF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \phi^*(X,S,r)= m_{\pi'}(X) + w(S,X)\bigl[r-m(X,S)\bigr] \;-\; V(\pi')

where:

- :math:`w(S,X)=\dfrac{\pi'(S\mid X)}{\pi_0(S\mid X)}` is the importance weight
- :math:`m_{\pi'}(X)=\mathbb{E}_{S\sim \pi'}[\,m(X,S)\,]` is the target policy expectation

Estimator
~~~~~~~~~

.. math::

   \hat{V}(\pi')=\frac{1}{n}\sum_{i=1}^n
   \Bigl[m_{\pi'}(X_i) + w_i\,\bigl(r_i-m(X_i,S_i)\bigr)\Bigr]

Cross-Fitting
~~~~~~~~~~~~~

Split data into K folds; train nuisance models on K−1 folds and evaluate on the hold-out fold so every row is scored by a model that never saw its reward. This yields √n-consistency and avoids over-fitting bias.

Theoretical Guarantees
~~~~~~~~~~~~~~~~~~~~~~

CJE's theoretical foundation rests on three key results that enable robust, efficient off-policy evaluation:

**Double Robustness**
   The estimator is unbiased if *either* the importance weights :math:`w(S,X)` **or** the outcome model :math:`m(X,S)` is correctly specified. This provides protection against model misspecification.

**Single-Rate Efficiency (Theorem 5.2)**
   Unlike classical doubly-robust estimators that require *both* nuisances to converge at the :math:`n^{-1/4}` rate, CJE achieves :math:`\sqrt{n}`-efficiency when only *one* nuisance function is well-specified. This is the **core theoretical innovation** of the CJE methodology.
   
   *Key insight*: Isotonic calibration of importance weights (ensuring :math:`\mathbb{E}[w] = 1`) restores single-rate efficiency even when the outcome model converges slowly.

**Semiparametric Efficiency Bound**
   When both nuisances are well-specified, CJE attains the **Cramér-Rao lower bound** - no regular unbiased estimator can achieve lower asymptotic variance.

   The efficient influence function:
   
   .. math::
   
      \phi^*(X,S,r) = m_{\pi'}(X) + w(S,X)[r-m(X,S)] - V(\pi')
   
   achieves the minimal possible variance :math:`\mathbb{E}[(\phi^*)^2]` in the nonparametric model.

**Valid Confidence Intervals**
   Plug-in HC3 variance estimation yields asymptotically valid confidence intervals: :math:`\hat{V} \pm 1.96\,\widehat{\mathrm{SE}}`

Calibration Methodology
~~~~~~~~~~~~~~~~~~~~~~~

CJE employs two types of isotonic calibration to achieve its theoretical guarantees:

**Judge Calibration**
   Raw judge scores :math:`s_{\text{raw}}` are mapped to calibrated rewards via isotonic regression:
   
   .. math::
   
      r = g_\phi(s_{\text{raw}})
   
   where :math:`g_\phi` is a monotonic function ensuring :math:`\mathbb{E}[r - Y] = 0` on oracle data.

**Weight Calibration**
   Importance weights are calibrated per cross-validation fold to ensure exact mean centering:
   
   .. math::
   
      w_{\text{cal}} = g_{\text{iso}}(w_{\text{raw}}) \quad \text{s.t.} \quad \mathbb{E}[w_{\text{cal}}] = 1
   
   This centering is **critical** for single-rate efficiency and removes finite-sample bias.

Implementation Enhancements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This implementation includes several robustness features beyond the baseline paper:

**Outcome Model Calibration**
   An optional third calibration step applies isotonic regression to outcome model predictions:
   
   .. math::
   
      \hat{m}_{\text{cal}}(X,S) = g_{\text{outcome}}(\hat{m}(X,S))
   
   This preserves ranking while correcting systematic prediction bias. *Enable with* ``calibrate_outcome=True`` *(default)*.

**Multi-Policy Joint Evaluation**
   Native support for :math:`K` target policies with full covariance estimation:
   
   .. math::
   
      \hat{\Sigma} = \frac{1}{n} \sum_i (\phi_i - \hat{V})(\phi_i - \hat{V})^\top \in \mathbb{R}^{K \times K}
   
   Enables statistically rigorous policy comparisons and portfolio optimization.

**Numerical Stabilization**
   Automatic log-ratio clipping, weight diagnostics, and ESS monitoring prevent numerical instabilities in production deployments.

Robustness Properties
~~~~~~~~~~~~~~~~~~~~

**Breakdown Point**
   The estimator remains stable even when a small fraction of importance weights become extreme (effective sample size monitoring provides early warning).

**Model Selection Robustness**
   Automatic outcome model selection based on sample size prevents overfitting in small-sample regimes.

**Teacher Forcing Consistency**
   Built-in diagnostics detect teacher forcing implementation bugs by monitoring identical policy weight consistency.

Convergence Rates
~~~~~~~~~~~~~~~~~

Under regularity conditions:

.. math::

   \sqrt{n}(\hat{V} - V) \xrightarrow{d} \mathcal{N}(0, \sigma^2_{\text{eff}})

where :math:`\sigma^2_{\text{eff}} = \mathbb{E}[(\phi^*)^2]` is the semiparametric efficiency bound.

**Single-rate regime**: If :math:`\|\hat{m} - m\|_2 = o_p(n^{-1/4})` **or** :math:`\|\hat{w} - w\|_2 = o_p(n^{-1/4})`, the convergence rate is preserved.

**Double-rate regime**: If both nuisances converge at :math:`n^{-1/4}`, the estimator achieves the efficiency bound.

Trajectory (MDP) Extension
~~~~~~~~~~~~~~~~~~~~~~~~~~

Everything above treats :math:`(X,S)` as a *single* sequence. For an agent we observe a trajectory

.. math::

   \tau = (H_0, A_1, H_1, A_2,\ldots, H_T, A_{T+1})

where :math:`H_t` is the dialogue / environment state before act :math:`A_{t+1}`. The logging policy factorises

.. math::

   \pi_0(\tau\mid X)=\prod_{t=0}^{T} \pi_0(A_{t+1}\mid H_t).

For any target policy :math:`\pi'` define the per-trajectory importance weight

.. math::

   W(\tau)=\prod_{t=0}^{T} \frac{\pi'(A_{t+1}\mid H_t)}{\pi_0(A_{t+1}\mid H_t)}.

Let the (possibly delayed) utility be

.. math::

   R(\tau)=\begin{cases}
     \sum_{t=0}^{T} \gamma^t\,r_t &\text{(per-step rewards)}\\
     Y_{\text{terminal}} &\text{(single final KPI)}
   \end{cases}

with :math:`\gamma\in(0,1]` a discount (we use :math:`\gamma=1` by default).

Outcome Model and EIF
^^^^^^^^^^^^^^^^^^^^^^

Define the outcome model :math:`m(H_t, A_{t+1})=\mathbb{E}[R\mid H_t,A_{t+1}]` and its target-policy expectation

.. math::

   m_{\pi'}(H_0)=\mathbb{E}_{\pi'}[m(H_t, A_{t+1})\mid H_0].

The efficient influence function now *telescopes* over steps (Jiang & Li, 2016):

.. math::

   \phi_{\text{traj}}^{(k)}(\tau,r)=\sum_{t=0}^{T} W_{1:t}^{(k)}\bigl[\gamma^{t}\,r_t-m(H_t,A_{t+1})\bigr]+m_{\pi_k'}(H_0)-V(\pi_k'),

where :math:`W_{1:t}^{(k)}` is the cumulative weight product up to step :math:`t` and :math:`\pi_k'` is the :math:`k^{\text{th}}` target policy.

Cross-fitting each trajectory as a whole and averaging :math:`\phi_{\text{traj}}` yields the *DR-CPO-MDP* estimator implemented in :class:`~cje.estimators.trajectory_drcpo.MultiDRCPOMDPEstimator`, which retains:

* **Double robustness** – unbiased if either the outcome model *or* any factor of :math:`\pi_0` is correct.
* **Semiparametric efficiency** – attains the Cramér–Rao lower bound when both nuisances converge at :math:`n^{-1/4}` rate.

Weight clipping, HC3 variance, and multiple-policy covariance carry over unchanged—simply replace per-sequence log-prob with the sum over steps.

Multiple-Policy Evaluation
---------------------------

For :math:`K` deterministic or stochastic target policies :math:`\{\pi^{(1)},\dots,\pi^{(K)}\}`:

* Compute a K-dimensional EIF vector:
  :math:`\phi^k(X,S,r)=m_{\pi^{(k)}}(X)+w^{(k)}(r-m)-V(\pi^{(k)})`

* Obtain joint point estimates :math:`\hat{V}\in\mathbb{R}^K`

* Estimate the full covariance matrix:
  :math:`\hat{\Sigma} = \frac{1}{n} \sum_i (\phi_i-\hat{V})(\phi_i-\hat{V})^\top`

* Pair-wise hypothesis tests (e.g., "Is :math:`\pi^i` better than :math:`\pi^j`?") use:
  :math:`\widehat{\mathrm{Var}}(\hat{V}_i-\hat{V}_j)=\hat{\Sigma}_{ii}+\hat{\Sigma}_{jj}-2\hat{\Sigma}_{ij}`

Practical Pipeline (Bird's-Eye View)
------------------------------------

.. code-block:: text

   1.  Logging           – collect (X,S, logprobs) with stochastic π₀
   2.  Cheap judge       – score each (X,S) once
   3.  Calibration       – align judge scores to true KPI on 20–30% oracle data
   4.  Outcome model     – cross-fitted regression m(X,S)
   5.  Target policies   – compute π'(S|X) for each policy (force-decode)
   6.  Estimation        – DR-CPO + HC3 variance, weight clip 20–50
   7.  Inference         – CIs, pair-wise Wald tests (Holm/BH-corrected)

Best-Practice Heuristics
-------------------------

.. list-table:: Configuration Recommendations
   :header-rows: 1
   :widths: 30 70

   * - Setting
     - Recommendation
   * - **Logging temperature**
     - ≥ 0.3 to ensure support overlap
   * - **Weight clipping**
     - 20–50; monitor clipped-mass < 2%
   * - **Outcome model**
     - Start small (ridge or tree-based); complexity only if CI coverage suffers
   * - **Judge drift**
     - Re-calibrate weekly or when KS distance > 0.05
   * - **Folds (cross-fit)**
     - K = 5 good default; K = 10 for ≤ 5k rows

Estimator Comparisons
---------------------

.. list-table:: Estimator Characteristics
   :header-rows: 1
   :widths: 15 15 15 30 25

   * - Estimator
     - Unbiased?
     - Double-robust?
     - Variance
     - Requires outcome model?
   * - IPS
     - ✔
     - ✖
     - High
     - No
   * - SNIPS
     - ≈✔
     - ✖
     - Medium
     - No
   * - **DR-CPO**
     - ✔
     - ✔
     - **Low (efficiency-bound)**
     - Yes
   * - MRDR
     - ✔
     - ✔
     - Can be lowest if weighted model fits very well
     - Yes, weighted

DR-CPO is usually preferred: flexible, robust to modest model error, and near-optimal variance.

Current Scope & Future Work
---------------------------

* **Implemented**: single-turn sequences, multi-turn **agent trajectories** (tool use), discrete or token actions, multiple target policies, full joint inference
* **Roadmap**: continuous action embeddings, adaptive exploration, formal drift detection

See Also
--------

* :doc:`../quickstart` – Get up and running quickly
* :doc:`../api/estimators` – Implementation details and usage examples
* :doc:`estimator_comparison` – Detailed estimator comparison
* :doc:`trajectory_methods` – Multi-step agent evaluation methods 