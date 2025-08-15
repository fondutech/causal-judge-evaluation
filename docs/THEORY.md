# Theory & Mathematical Foundations

## Pipeline Overview

```
[LOGS (π₀)]                                   [ORACLE SLICE (~1-5%)]
• X: context/prompt                           • S: judge score
• A: completion from π₀                       • Y: human label (truth)
• S: judge score                              • Same judge config
• log p_π₀(A|X)                               
     │                                               │
     └──────────────────────────┬────────────────────┘
                                │
                                ▼
                ┌───────────────────────────────────────┐
                │ REWARD CALIBRATION (Isotonic f)       │
                │ • Monotone, mean-preserving (PAV)     │
                │ • Apply R=f(S) to all logs           │
                │ • Targets E[Y(π′)] if (J2-M) holds   │
                └───────────────────────────────────────┘
                                │
                                ▼
                ┌───────────────────────────────────────┐
                │ RAW WEIGHTS W_π′ (TF Cache)           │
                │ • W_π′ = exp(log p_π′ - log p_π₀)    │
                │ • Normalize to mean one (Hájek)       │
                └───────────────────────────────────────┘
                                │
                                ▼
                ┌───────────────────────────────────────┐
                │ SIMCAL: WEIGHT CALIBRATION            │
                │ • Monotone projection indexed by S    │
                │ • OOF stacking to minimize variance   │
                │ • Variance cap ρ: blend → reproject  │
                └───────────────────────────────────────┘
                        │                   │
        ┌───────────────┘                   └────────────────┐
        ▼                                                     ▼
┌─────────────────────┐                      ┌──────────────────────────┐
│ CAL-IPS             │                      │ DR-CPO / TMLE            │
│ V̂ = mean(W_c × R)   │                      │ V̂ = ĝ + W_c×(R−q̂_b)     │
└─────────────────────┘                      └──────────────────────────┘
                        │                   │
                        └─────────┬─────────┘
                                  ▼
                    ┌───────────────────────────┐
                    │ DIAGNOSTICS & GATES       │
                    │ • ESS, Hill index         │
                    │ • Judge reliability       │
                    │ • Orthogonality tests     │
                    └───────────────────────────┘
```

## Notation & Terminology

### Data & Policies
- **π₀**: Logging/base policy (what generated your data)
- **π′**: Target policy (what you want to evaluate)
- **X**: Context/prompt input
- **A**: Completion/response from the policy
- **S**: Judge score ∈ [0,1] (from automatic evaluator like GPT-4)
- **Y**: Oracle label ∈ [0,1] (human ground truth)
- **R**: Calibrated reward ∈ [0,1] (isotonic transformation of S)

### Transformations & Weights
- **f**: Isotonic calibration function (S → R mapping)
- **p_π(A|X)**: Probability of response A under policy π
- **W_π′**: Raw importance weights = exp(log p_π′(A|X) - log p_π₀(A|X))
- **W_c**: Calibrated weights (after SIMCal transformation)
- **ρ**: Variance cap parameter (controls weight stability)

### Estimation
- **V̂**: Estimated policy value E[R(π′)]
- **ĝ_bπ′(X)**: Marginalized outcome model E[R|X] under π′
- **q̂_b(X,A)**: Baseline outcome model E[R|X,A]
- **mean(·)**: Empirical average over samples

### Algorithms & Methods
- **PAV**: Pool-Adjacent-Violators (isotonic regression algorithm)
- **SIMCal**: Surrogate-Indexed Monotone Calibration
- **OOF**: Out-of-fold cross-fitting (prevents overfitting)
- **TF Cache**: Teacher Forcing cache (stores log probabilities)
- **Hájek**: Self-normalized importance sampling (ensures E[W]=1)

### Diagnostics
- **ESS**: Effective Sample Size = (Σw)²/Σw² (overlap quality)
- **Hill index**: Tail heaviness measure (should be ≥ 2)
- **CI**: Confidence interval (typically 95%)

## Core Assumptions

### Logging Assumptions (D1-D3)
- **(D1)**: Fixed logger - π₀ doesn't change during data collection
- **(D2)**: Overlap - all actions have p_π₀(A|X) > 0
- **(D3)**: I.I.D. sampling from logging distribution

### Judge Assumptions (J1-J2)
- **(J1)**: I.I.D. oracle slice - random subset for calibration
- **(J2-M)**: Judge monotonicity - higher S → higher E[Y|S]

### Regularity Conditions (R1-R3)
- **(R1)**: Bounded second moments of weights
- **(R2)**: Lipschitz continuity of outcome models
- **(R3)**: Rate conditions - nuisance functions converge at n^(-1/4)

## Key Theoretical Results

Under the above assumptions:

### Mean Preservation Theorem
Isotonic calibration preserves the population mean exactly:
```
E[f(S)] = E[Y]
```
where f is the isotonic regression function fitted on the oracle slice.

### Variance Dominance Theorem
SIMCal weakly reduces variance by majorization:
```
Var(W_c) ≤ Var(W_raw)
```
with equality only when raw weights are already monotone in S.

### √n-Normality
Both Cal-IPS and DR-CPO (Doubly-Robust Counterfactual Policy Optimization) achieve asymptotic normality:
```
√n(V̂ - V(π′)) → N(0, σ²)
```

### Semiparametric Efficiency
DR-CPO (Doubly-Robust Counterfactual Policy Optimization) attains the semiparametric efficiency bound when both nuisance functions are consistently estimated.

## Mathematical Details

### SIMCal Algorithm

1. **Input**: Raw weights W, surrogate index S, variance cap ρ
2. **Sort** by S: (W_σ, S_σ) where σ is sorting permutation
3. **Project** onto monotone functions via PAV
4. **Blend** with baseline: W_blend = αW_monotone + (1-α)·1
5. **Reproject** to maintain monotonicity
6. **Output**: Calibrated weights W_c

### Isotonic Regression (PAV)

The Pool-Adjacent-Violators algorithm finds:
```
f* = argmin_f Σ(Y_i - f(S_i))² 
     subject to: f monotone increasing
```

### Doubly Robust Score

The influence function for DR estimation:
```
ψ(X,A,R) = ĝ(X) + W(A|X)·(R - q̂(X,A))
```

This achieves double robustness: consistent if either ĝ or q̂ is correct.

## Advanced Topics

### Cross-Fitting for DR

To avoid overfitting bias, we use K-fold cross-fitting:
1. Split data into K folds
2. For each fold k:
   - Fit nuisance functions on other K-1 folds
   - Evaluate on fold k
3. Combine all fold estimates

### Targeted Maximum Likelihood (TMLE)

TMLE iteratively updates the outcome model to minimize bias:
1. Initial fit: q⁰(X,A)
2. Targeting step: logit(q¹) = logit(q⁰) + εW
3. Update until convergence

### Oracle Coverage Sensitivity

The oracle coverage parameter controls the bias-variance tradeoff:
- **Low coverage (1-5%)**: Higher variance in calibration, lower annotation cost
- **High coverage (25-50%)**: Lower variance, higher cost
- **Optimal**: Typically 5-10% for most applications

## References

- Dudík, Langford, and Li (2011): Doubly-Robust Counterfactual Policy Optimization (DR-CPO)
- Dudík et al. (2014): Doubly robust policy evaluation
- Jiang, Li, Kallus, et al. (2019): More Robust Doubly-Robust (MRDR) estimator
- van der Laan et al. (2025): Isotonic calibration for causal inference
- van der Laan & Rubin (2006): Targeted Maximum Likelihood Estimation (TMLE)
- Chernozhukov et al. (2018): Double/debiased machine learning
- Our paper: [Forthcoming]