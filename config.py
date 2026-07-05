import os

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
DATA_REPO   = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-active-inference-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}

MACRO_COLS_CORE     = ["VIX", "DXY", "T10Y2Y"]
MACRO_COLS_EXTENDED = ["IG_SPREAD", "HY_SPREAD"]

# ── Rolling windows (trading days) ────────────────────────────────────────────
WINDOWS = [63, 126, 252, 504]

# ── Predictive Coding / Active Inference hyperparameters ──────────────────────
# Karl Friston's variational free energy principle: a generative model
# maintains beliefs q(s) about hidden states s and minimizes
#
#     F = KL[q(s) || p(s)]  -  E_q[log p(o|s)]
#       = complexity        -  accuracy
#
# which upper-bounds surprise (-log p(o)) of an observation o. Under Gaussian
# assumptions this reduces EXACTLY to a hierarchical Kalman filter — the
# standard practical implementation of predictive coding:
#
#   Level 2 (slow "belief"):  mu2_t = mu2_{t-1} + w_t      (random-walk drift belief)
#   Level 1 (fast, observed): r_t   = mu2_t + v_t            (observation = belief + noise)
#
# At each step the model predicts today's return from its current belief,
# compares against what actually happened (the prediction error), and
# updates its belief by precision-weighting that error — literally
# minimizing free energy step by step, not fit after the fact.
#
# CORE THESIS: a return that is highly SURPRISING relative to the model's
# own current beliefs (large precision-weighted prediction error, i.e. high
# free energy) is treated as anomalous — and anomalous moves are read as
# CONTRARIAN, favoring reversion back toward the model's belief, not
# confirmation of the move. This is the opposite structure from the other
# engines in this suite, where a secondary diagnostic CONFIRMS the primary
# forecast's direction; here the surprise-driven component explicitly
# argues AGAINST the most recent move.
#
# Pure univariate, no macro conditioning — belief updating operates on the
# ticker's own return series, matching the design of N-HiTS and EDMD
# elsewhere in this suite (architecture/theory-driven, not exogenously
# conditioned).

Q_R_RATIO = 0.05    # process noise / observation noise ratio: how "sticky" (small)
                     # vs. "plastic" (large) the slow belief is. Observation noise R
                     # is estimated directly from the window's own return variance;
                     # process noise Q = R * Q_R_RATIO.

SURPRISE_ZSCORE_CAP = 3.0   # cap on the surprise-driven contrarian signal, so a
                             # single extreme outlier day can't dominate the score

PRED_HORIZON = 21    # kept for documentation/consistency; under the model's own
                      # random-walk belief assumption the point forecast for the
                      # drift is horizon-invariant (a martingale) — there is no
                      # separate multi-step path to compute, unlike the ML engines

# ── Score construction ────────────────────────────────────────────────────────
# surprise_signal  : -clip(zscore(today's prediction error vs. this window's own
#                    error history), +/-CAP) — the PRIMARY, contrarian component:
#                    an anomalously large positive surprise argues for reversion
#                    DOWN, an anomalously large negative surprise argues for
#                    reversion UP
# belief_signal    : zscore of the current drift belief relative to its own
#                    trajectory within the window — a secondary, non-contrarian
#                    "how extreme is the model's current belief itself" context
# model_confidence : 1 / (1 + max(0, excess kurtosis of standardized prediction
#                    errors)) — the Gaussian generative model's own residuals
#                    should be approximately Gaussian if well-specified; fat
#                    tails (occasional huge surprises) mean less trust in the
#                    whole apparatus, including the surprise signal itself

WEIGHT_SURPRISE    = 0.50
WEIGHT_BELIEF       = 0.25
WEIGHT_CONFIDENCE   = 0.25

TOP_N = 3
