# Estimator Implementation Notes

## Doubly Robust Estimators (DR-CPO / MRDR)

### Why a target-policy draw is needed

Both calibrated **DR-CPO** and **MRDR** rely on the identity:

$$
\psi_i = \underbrace{\mathbb{E}_{S\sim\pi'}[m(X_i,S)]}_{\text{baseline}} + W_i\bigl(r_i - m(X_i,S_i)\bigr)
$$

where:
- $m(x,s) = \mathbb{E}[r \mid X=x, S=s]$ is the outcome surface
- $r_i$ is the calibrated reward for the *logged* sequence $S_i$  
- $W_i$ is the calibrated weight $\pi'(S_i|X_i)/\pi_0(S_i|X_i)$

The **baseline term** (the expectation under $\pi'$) is what cancels first-order weight noise and gives DR its variance advantage. Without an estimate of that term, the score collapses to IPS and MRDR's control-variate coefficient cannot be computed.

### One Monte-Carlo draw is enough

- Drawing **one** continuation $S_i^{(\pi')} \sim \pi'(\cdot|X_i)$ per context and evaluating $m(X_i,S_i^{(\pi')})$ yields an *unbiased* plug-in estimate of the baseline
- By Neyman orthogonality, the extra Monte-Carlo noise only enters at **second order**; $\sqrt{n}$ efficiency is preserved
- Computationally, one short generation is negligible compared with the teacher-forced pass already performed to obtain $W_i$

### Role of the cheap judge score

To score that single target completion you also need a **cheap judge evaluation** so that its reward is on the same calibrated scale as the logged rows:

$$
r^{(\pi')}_i = g_\phi\bigl(s_{\text{raw}}^{(\pi')}\bigr)
$$

where $g_\phi$ is the isotonic map learned on the oracle slice. This keeps $m(X_i,S_i^{(\pi')})$ and $r_i$ comparable, ensuring the DR control term remains centered.

### Practical implications for CJE

**Required**: At least **one target completion + judge score per context** when the estimator is `dr` or `mrdr`.

**Default behavior**: The library should either:
- Automatically downgrade to **calibrated IPW/SNIPW** (unbiased but wider CIs), OR  
- Raise a clear message: *"DR/MRDR needs ≥1 target draw; set `generate_target=1` or choose `ipw`/`snipw`"*

This guarantee protects users from accidentally running a high-variance IPW while believing they are getting DR's variance reduction.

## IPW and SNIPW Estimators

### Standard IPW (Inverse Propensity Weighting)

The simplest off-policy estimator:

$$
\hat{V}_{\text{IPW}} = \frac{1}{n} \sum_{i=1}^n W_i \cdot r_i
$$

**Pros**:
- Unbiased
- No target policy samples needed
- Simple to implement

**Cons**:
- High variance when weights are extreme
- No variance reduction from control variates

### SNIPW (Self-Normalized IPW)

Normalizes by the sum of weights:

$$
\hat{V}_{\text{SNIPW}} = \frac{\sum_{i=1}^n W_i \cdot r_i}{\sum_{i=1}^n W_i}
$$

**Pros**:
- More stable than IPW when weights vary
- Still no target samples needed
- Often lower variance in practice

**Cons**:
- Slightly biased (but consistent)
- Bias decreases as $n \to \infty$

## Choosing an Estimator

### Use IPW/SNIPW when:
- You cannot generate target policy samples
- You need strictly unbiased estimates (IPW only)
- Computational budget is very tight

### Use DR/MRDR when:
- You can generate at least one target sample per prompt
- Variance reduction is important
- You have judge capacity for scoring target samples

### Arena 10K Experiment Setup

In this experiment, we test all estimators:
- **IPW**: Baseline unbiased estimator
- **SNIPW**: Self-normalized variant
- **DR**: Doubly robust (requires target samples)

Note: For DR to work properly in this experiment, we need target policy continuations which are not currently in the dataset. The experiment may need modification to properly test DR estimators.