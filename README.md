# 🧠 P2-ETF-ACTIVE-INFERENCE

**Predictive Coding / Active Inference Engine — Karl Friston**

Part of the **P2Quant Engine Suite** · [P2SAMAPA](https://github.com/P2SAMAPA)

---

## What This Engine Does

This engine runs a genuine hierarchical Bayesian belief-updating process
per ETF — Karl Friston's variational free energy principle, implemented
via its exact Gaussian special case: a two-level Kalman filter. Unlike
every other engine in this suite, the **secondary diagnostic argues
against the primary move, not with it** — a highly surprising return is
treated as anomalous and read as a contrarian mean-reversion signal, not
confirmation of a trend.

---

## Theory

### Variational Free Energy

A generative model maintains approximate beliefs `q(s)` about hidden
states `s` and minimizes:

```
F = KL[q(s) || p(s)]  -  E_q[log p(o|s)]
  = complexity          -  accuracy
```

which upper-bounds surprise (`-log p(o)`) of an observation, since
`KL[q(s)||p(s|o)] >= 0` implies `F >= -log p(o)`.

### Predictive Coding as a Hierarchical Kalman Filter

Under Gaussian assumptions, free energy minimization reduces EXACTLY to a
two-level local-level model:

```
Level 2 (slow "belief"):   mu2_t = mu2_{t-1} + w_t,   w_t ~ N(0, Q)
Level 1 (fast, observed):  r_t   = mu2_t + v_t,        v_t ~ N(0, R)
```

At each step:

```
1. PREDICT : mu_prior = mu2_{t-1|t-1},  var_prior = var2_{t-1|t-1} + Q
2. COMPARE : epsilon_t = r_t - mu_prior                (prediction error)
3. UPDATE  : K_t = var_prior / (var_prior + R)          (precision-weighted gain)
             mu2_{t|t}  = mu_prior + K_t * epsilon_t
             var2_{t|t} = (1 - K_t) * var_prior
```

This **is** free energy minimization step by step — not a fit performed
after the fact. Observation noise `R` is estimated from the window's own
return variance; process noise `Q = R * Q_R_RATIO` controls how "sticky"
vs. "plastic" the slow belief is.

### Free Energy, Computed via Its Actual Decomposition

```
precision_t   = 1 / (var_prior + R)
accuracy_t    = 0.5 * precision_t * epsilon_t^2 + 0.5*log(2*pi/precision_t)
complexity_t  = KL[ N(mu2_t|t, var2_t|t) || N(mu_prior, var_prior) ]
F_t           = accuracy_t + complexity_t
```

### Core Thesis

A return that is highly surprising relative to the model's own current
beliefs is anomalous — and anomalous moves are read as **contrarian**,
favoring reversion back toward belief rather than confirming the move.
This is the opposite structure from most other engines in this suite: here
the secondary signal explicitly argues against the primary move.

### Score Construction

```
score = 0.50*surprise_signal + 0.25*belief_signal + 0.25*model_confidence
```

| Component | Meaning |
|-----------|---------|
| surprise_signal | -clip(zscore of today's prediction error vs. this window's own error history) — the primary, contrarian driver |
| belief_signal | zscore of the current drift belief relative to its own trajectory — secondary, non-contrarian context |
| model_confidence | 1/(1+max(0, excess kurtosis of standardized errors)) — validity check on the Gaussian assumption itself |

### Validation

Validated on synthetic data with known ground truth before shipping:
- **KL divergence formula** verified against hand-calculated special cases
- **Belief tracking**: filter correctly tracks a synthetic process generated from exactly its own assumed model (correlation 0.75 between filtered and true belief)
- **model_confidence discriminates correctly**: 0.63 for a well-specified Gaussian process vs. 0.09 for a genuinely fat-tailed (misspecified) process with occasional large jumps
- **surprise_signal direction**: confirmed correctly contrarian in both directions — a huge positive surprise spike produces `surprise_signal = -3.0` (max negative, predicting reversion down); a huge negative spike produces `+3.0` (predicting reversion up)

---

## Distinction from Other Information-Theoretic Engines in the Suite

| Engine | Core object | Belief updating over time? |
|--------|-------------|------------------------------|
| ENTROPY-ENGINE | Shannon entropy of the return distribution | No — static distributional measure |
| INFO-BOTTLENECK | Mutual information (compression vs. relevance) | No — static representation measure |
| THERMO-BOTTLENECK | Thermodynamic free energy analogy (F=U-TS) | No — physical/statistical-mechanical analogy |
| **This engine** | **Variational (Bayesian) free energy** | **Yes — genuine sequential posterior updating** |

"Free energy" is a genuinely overloaded term: thermodynamic free energy
(THERMO-BOTTLENECK) and variational free energy (this engine) share a
historical mathematical analogy but are different objects with different
interpretations. This engine is the only one of the four built around an
explicit hierarchical generative model whose beliefs are actually
*updated*, observation by observation, via precision-weighted prediction
errors — not a static information-theoretic summary of a distribution.

---

## Universes & Windows

| Universe | Tickers |
|---|---|
| FI_COMMODITIES | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| EQUITY_SECTORS | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM, IWD, IWO, XLB, XLRE |
| COMBINED | All of the above |

**Windows:** `63d · 126d · 252d · 504d`

---

## Repository Structure

```
P2-ETF-ACTIVE-INFERENCE/
├── config.py                    # Universes, filter hyperparameters, score weights
├── data_manager.py               # HuggingFace loader
├── active_inference_engine.py     # Core: hierarchical Kalman filter, free energy, surprise signal
├── trainer.py                     # Orchestrator
├── push_results.py                # HfApi.upload_file wrapper
├── streamlit_app.py                # Two-tab Streamlit dashboard
├── us_calendar.py                 # US trading calendar helper
├── requirements.txt
└── .github/
    └── workflows/
        └── daily.yml               # Single job
```

---

## Setup

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-ACTIVE-INFERENCE
cd P2-ETF-ACTIVE-INFERENCE
pip install -r requirements.txt

export HF_TOKEN=hf_...
python trainer.py
streamlit run streamlit_app.py
```

**Required GitHub secret:** `HF_TOKEN`

**Required HuggingFace dataset repo:** `P2SAMAPA/p2-etf-active-inference-results`

---

## References

- Friston, K. (2010). The Free-Energy Principle: A Unified Brain Theory?
  Nature Reviews Neuroscience.
- Friston, K., Kilner, J. & Harrison, L. (2006). A Free Energy Principle
  for the Brain. Journal of Physiology-Paris.
- Rao, R. & Ballard, D. (1999). Predictive Coding in the Visual Cortex.
  Nature Neuroscience.
