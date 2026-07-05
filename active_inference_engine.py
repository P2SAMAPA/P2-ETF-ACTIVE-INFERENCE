"""
active_inference_engine.py — Predictive Coding / Active Inference Engine
==============================================================================

Theory
------
**Variational free energy.** Karl Friston's free energy principle treats a
generative model as maintaining approximate beliefs q(s) about hidden
states s, and minimizing:

    F = KL[q(s) || p(s)]  -  E_q[log p(o|s)]
      = complexity          -  accuracy

which upper-bounds surprise (-log p(o)) of an observation o (since
KL[q(s)||p(s|o)] >= 0 implies F >= -log p(o)). Minimizing F is therefore
minimizing an upper bound on how surprising observations are, given the
model's own beliefs.

**Predictive coding as a hierarchical Kalman filter.** Under Gaussian
assumptions, this reduces EXACTLY to a two-level local-level model:

    Level 2 (slow "belief"):   mu2_t = mu2_{t-1} + w_t,   w_t ~ N(0, Q)
    Level 1 (fast, observed):  r_t   = mu2_t + v_t,        v_t ~ N(0, R)

At each step:
    1. PREDICT   : mu_prior = mu2_{t-1|t-1},  var_prior = var2_{t-1|t-1} + Q
    2. COMPARE   : epsilon_t = r_t - mu_prior          (prediction error)
    3. UPDATE    : K_t = var_prior / (var_prior + R)   (precision-weighted gain)
                   mu2_{t|t}  = mu_prior + K_t * epsilon_t
                   var2_{t|t} = (1 - K_t) * var_prior

This IS free energy minimization step by step, not a fit performed after
the fact — steps 1-3 are literally gradient descent on F under Gaussian
assumptions (Friston has written about this equivalence explicitly).

**Free energy per step**, computed via its actual decomposition:

    precision_t   = 1 / (var_prior + R)
    accuracy_t    = 0.5 * precision_t * epsilon_t^2 + 0.5*log(2*pi/precision_t)
    complexity_t  = KL[ N(mu2_t|t, var2_t|t) || N(mu_prior, var_prior) ]
    F_t           = accuracy_t + complexity_t

**Core thesis.** A return that is highly surprising relative to the
model's own current beliefs (large precision-weighted prediction error,
high F) is treated as anomalous. Anomalous moves are read as CONTRARIAN —
favoring reversion back toward the belief, not confirmation of the move.
This is the opposite structure from most other engines in this suite: here
the secondary signal explicitly argues AGAINST the primary move rather
than confirming it.

**Score construction**

    score = 0.50*surprise_signal + 0.25*belief_signal + 0.25*model_confidence

| Component        | Meaning                                                                 |
|--------------------|----------------------------------------------------------------------------|
| surprise_signal    | -clip(zscore of today's prediction error vs. this window's own error history) — the contrarian core signal |
| belief_signal      | zscore of the current drift belief relative to its own trajectory — secondary, non-contrarian context |
| model_confidence   | 1/(1+max(0, excess kurtosis of standardized errors)) — validity check on the Gaussian assumption itself |

**Distinction from other information-theoretic engines in this suite**

| Engine              | Core object                          | Belief updating over time? |
|----------------------|----------------------------------------|-------------------------------|
| ENTROPY-ENGINE        | Shannon entropy of the return distribution | No — static distributional measure |
| INFO-BOTTLENECK       | Mutual information (compression vs. relevance) | No — static representation measure |
| THERMO-BOTTLENECK     | Thermodynamic free energy analogy (F=U-TS) | No — physical/statistical-mechanical analogy |
| **This engine**       | **Variational (Bayesian) free energy** | **Yes — genuine sequential posterior updating** |

"Free energy" is a genuinely overloaded term: thermodynamic free energy
(THERMO-BOTTLENECK) and variational free energy (this engine) share a
historical mathematical analogy but are different objects with different
interpretations. This engine is the only one of the four built around an
explicit hierarchical generative model with beliefs that are actually
updated, observation by observation, via precision-weighted prediction
errors — not a static information-theoretic summary of a distribution.

References
----------
- Friston, K. (2010). The Free-Energy Principle: A Unified Brain Theory?
  Nature Reviews Neuroscience.
- Friston, K., Kilner, J. & Harrison, L. (2006). A Free Energy Principle
  for the Brain. Journal of Physiology-Paris.
- Rao, R. & Ballard, D. (1999). Predictive Coding in the Visual Cortex.
  Nature Neuroscience.
"""

import numpy as np
import pandas as pd
from typing import List

import config


def kl_gaussian(mu_a: float, var_a: float, mu_b: float, var_b: float) -> float:
    """KL[N(mu_a,var_a) || N(mu_b,var_b)], the exact univariate Gaussian formula."""
    return float(0.5 * (np.log(var_b / var_a) + (var_a + (mu_a - mu_b) ** 2) / var_b - 1.0))


def run_predictive_coding_filter(log_ret: np.ndarray, R: float, Q: float) -> dict:
    """
    Hierarchical Kalman / predictive-coding filter over a return series.
    Returns per-timestep arrays: mu_prior, var_prior, mu_post, var_post,
    epsilon (prediction error), precision, F (free energy).
    """
    T = len(log_ret)
    mu_post, var_post = 0.0, R

    mu_priors  = np.zeros(T)
    var_priors = np.zeros(T)
    mu_posts   = np.zeros(T)
    var_posts  = np.zeros(T)
    epsilons   = np.zeros(T)
    precisions = np.zeros(T)
    F_vals     = np.zeros(T)

    for t in range(T):
        mu_prior  = mu_post
        var_prior = var_post + Q

        r_t = log_ret[t]
        eps = r_t - mu_prior
        precision = 1.0 / (var_prior + R)

        K = var_prior / (var_prior + R)
        mu_post_new  = mu_prior + K * eps
        var_post_new = max((1.0 - K) * var_prior, 1e-12)

        accuracy   = 0.5 * precision * eps ** 2 + 0.5 * np.log(2 * np.pi / precision)
        complexity = kl_gaussian(mu_post_new, var_post_new, mu_prior, var_prior)
        F_t = accuracy + complexity

        mu_priors[t], var_priors[t] = mu_prior, var_prior
        mu_posts[t], var_posts[t]   = mu_post_new, var_post_new
        epsilons[t], precisions[t]  = eps, precision
        F_vals[t] = F_t

        mu_post, var_post = mu_post_new, var_post_new

    return {
        "mu_prior": mu_priors, "var_prior": var_priors,
        "mu_post": mu_posts, "var_post": var_posts,
        "epsilon": epsilons, "precision": precisions, "F": F_vals,
    }


def excess_kurtosis(x: np.ndarray) -> float:
    """Population excess kurtosis via method of moments (Gaussian => 0)."""
    mu = x.mean()
    sd = x.std() + 1e-12
    m4 = np.mean((x - mu) ** 4)
    return float(m4 / (sd ** 4) - 3.0)


# ── Per-ticker forecast + diagnostics ───────────────────────────────────────────

def forecast_and_diagnose(prices: pd.DataFrame, ticker: str, window: int, rng: np.random.Generator = None):
    """
    Run the predictive-coding filter for one ticker using only data present
    in `prices` (the caller controls the as-of cutoff simply by how much
    history `prices` contains) and return diagnostics for the belief state
    as of the LAST row of `prices`. Returns None on failure.

    This is the single source of truth for per-ticker active-inference
    scoring. `rng` is accepted (unused) for interface consistency with the
    other engines in this suite.
    """
    ps = prices[ticker].dropna()
    if len(ps) < window + 15:
        return None

    log_ret_full = np.log(ps / ps.shift(1)).dropna().values
    log_ret = log_ret_full[-window:]
    if len(log_ret) < 20:
        return None

    R = float(np.var(log_ret) + 1e-12)     # observation noise: empirical return variance
    Q = R * config.Q_R_RATIO                # process noise: fraction of observation noise

    result = run_predictive_coding_filter(log_ret, R, Q)

    # ── belief_forecast: current posterior belief about the slow drift ────────
    belief_forecast = float(result["mu_post"][-1])
    belief_signal = float(
        (result["mu_post"][-1] - result["mu_post"].mean()) / (result["mu_post"].std() + 1e-12)
    )

    # ── surprise_signal: how anomalous was TODAY's prediction error ───────────
    eps_hist = result["epsilon"]
    eps_today = eps_hist[-1]
    eps_mu, eps_sd = eps_hist.mean(), eps_hist.std() + 1e-12
    z = (eps_today - eps_mu) / eps_sd
    z_capped = float(np.clip(z, -config.SURPRISE_ZSCORE_CAP, config.SURPRISE_ZSCORE_CAP))
    surprise_signal = -z_capped   # contrarian: big positive surprise -> reversion down

    # ── model_confidence: are standardized residuals approximately Gaussian? ──
    standardized_eps = result["epsilon"] / np.sqrt(result["var_prior"] + R)
    ek = excess_kurtosis(standardized_eps)
    model_confidence = float(1.0 / (1.0 + max(0.0, ek)))

    avg_free_energy = float(result["F"].mean())

    return {
        "belief_forecast": belief_forecast,
        "surprise_signal": surprise_signal,
        "belief_signal": belief_signal,
        "model_confidence": model_confidence,
        "prediction_error_today": float(eps_today),
        "avg_free_energy": avg_free_energy,
    }


# ── Main scoring function ─────────────────────────────────────────────────────

def compute_active_inference_scores(
    prices:    pd.DataFrame,
    macro_df:  pd.DataFrame,
    tickers:   List[str],
    window:    int,
) -> pd.DataFrame:
    """
    Run the predictive-coding filter per ETF (pure univariate belief
    updating — no macro conditioning) and extract a surprise-driven
    contrarian signal. Returns a DataFrame of score + diagnostics
    (cross-sectional z-scored on the composite).
    """
    cols = ["score", "belief_forecast", "surprise_signal", "belief_signal",
            "model_confidence", "prediction_error_today"]
    avail = [t for t in tickers if t in prices.columns]
    if not avail:
        return pd.DataFrame(columns=cols)

    raw_scores = {}

    for ticker in avail:
        print(f"    Filtering beliefs for {ticker}")
        diag = forecast_and_diagnose(prices, ticker, window)
        if diag is None:
            continue

        print(f"    {ticker}: surprise={diag['surprise_signal']:.3f}  "
              f"belief={diag['belief_signal']:.3f}  confidence={diag['model_confidence']:.3f}")

        composite = (
            config.WEIGHT_SURPRISE   * diag["surprise_signal"]
            + config.WEIGHT_BELIEF     * diag["belief_signal"]
            + config.WEIGHT_CONFIDENCE * diag["model_confidence"]
        )
        raw_scores[ticker] = {"composite": composite, **diag}

    if not raw_scores:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(raw_scores).T
    mu_s, std_s = df["composite"].mean(), df["composite"].std()
    if std_s < 1e-10:
        df["score"] = 0.0
    else:
        df["score"] = (df["composite"] - mu_s) / std_s
    return df[cols]
