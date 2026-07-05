import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Active Inference Engine", layout="wide")

st.markdown("""
<style>
.main-header { font-size:2.4rem; font-weight:700; color:#2b1d3e; margin-bottom:0.3rem; }
.sub-header  { font-size:1.1rem; color:#555; margin-bottom:1.5rem; }
.uni-title   { font-size:1.4rem; font-weight:600; margin-top:1rem; margin-bottom:0.8rem;
               padding-left:0.5rem; border-left:5px solid #7b5ea7; }
.etf-card    { background:linear-gradient(135deg,#2b1d3e 0%,#7b5ea7 100%); color:white;
               border-radius:14px; padding:1rem; margin:0.4rem; text-align:center;
               box-shadow:0 4px 6px rgba(0,0,0,0.2); }
.win-card    { background:linear-gradient(135deg,#2b1d3e 0%,#4a3766 100%); color:white;
               border-radius:14px; padding:1rem; margin:0.4rem; text-align:center;
               box-shadow:0 4px 6px rgba(0,0,0,0.2); }
.etf-ticker  { font-size:1.3rem; font-weight:bold; }
.etf-score   { font-size:0.88rem; margin-top:0.25rem; opacity:0.9; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧠 Active Inference Engine</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Karl Friston — Predictive Coding / Variational Free Energy · '
    'Hierarchical Kalman belief updating, precision-weighted prediction error · '
    'Surprise-driven CONTRARIAN signal (anomalous moves → reversion) · '
    'Multi-window cross-sectional z-score</div>',
    unsafe_allow_html=True)

st.sidebar.markdown("## Active Inference Engine")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown(f"**Windows:** {config.WINDOWS}")
st.sidebar.markdown(f"**Q/R ratio:** {config.Q_R_RATIO} (belief plasticity)")
st.sidebar.markdown(f"**Surprise clip:** ±{config.SURPRISE_ZSCORE_CAP} z-score")
st.sidebar.markdown(
    f"**Weights:** Surprise {config.WEIGHT_SURPRISE:.0%} | "
    f"Belief {config.WEIGHT_BELIEF:.0%} | "
    f"Confidence {config.WEIGHT_CONFIDENCE:.0%}")

HF_TOKEN    = config.HF_TOKEN
OUTPUT_REPO = config.OUTPUT_REPO


@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        files = [f["name"] for f in fs.ls(f"datasets/{OUTPUT_REPO}",
                                           detail=True, recursive=True)
                 if f["type"] == "file"]
        return files, None
    except Exception as e:
        return [], str(e)


def find_latest(files, prefix):
    matches = sorted([f for f in files if f.endswith(".json") and prefix in f],
                     reverse=True)
    return matches[0] if matches else None


@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


files, list_error = list_repo_files()

with st.expander("🔧 Debug: what the dashboard sees on HuggingFace", expanded=bool(list_error)):
    st.markdown(f"**Repo:** `{OUTPUT_REPO}`  ·  **Token set:** {'yes' if bool(HF_TOKEN) else 'no'}")
    if list_error:
        st.error(f"Could not list repo files: {list_error}")
        st.markdown(
            "If this is a permissions/auth error and `p2-etf-active-inference-results` "
            "is a **private** HF dataset, this Streamlit environment needs its own "
            "`HF_TOKEN` — separate from whatever secret your GitHub Actions workflow "
            "uses. Setting it for Actions does not set it here."
        )
    else:
        st.write(f"{len(files)} file(s) found:")
        st.code("\n".join(sorted(files)) if files else "(empty)")

tab1_path = find_latest(files, "active_inference_engine_2")
tab2_path = find_latest(files, "active_inference_engine_windows_")

if not tab1_path:
    if list_error:
        st.error(
            "Could not reach HuggingFace to look for results (see 🔧 Debug above) "
            "— this is not the same as 'trainer hasn't run yet'."
        )
    else:
        st.error(
            "Connected to HuggingFace successfully, but no file matching "
            "`active_inference_engine_2*.json` was found (see 🔧 Debug above for "
            "the exact file list). Check the filename trainer.py actually pushed."
        )
    st.stop()

data1 = load_json(tab1_path)
if "error" in data1:
    st.error(f"Error loading data: {data1['error']}")
    st.stop()

data2      = load_json(tab2_path) if tab2_path else None
universes1 = data1["universes"]
universes2 = data2["universes"] if data2 and "error" not in data2 else None

st.sidebar.markdown(f"**Run date:** `{data1.get('run_date','?')}`")

tab1, tab2 = st.tabs(["🏆 Best Window per ETF", "🔍 Explore by Window"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🏆 Top ETFs — Surprise-Driven Contrarian Signal")

    with st.expander("Active Inference Methodology", expanded=True):
        st.markdown("""
Karl Friston's **variational free energy** principle: a generative model
maintains beliefs and minimizes

```
F = KL[q(s)||p(s)] - E_q[log p(o|s)]  =  complexity - accuracy
```

Under Gaussian assumptions this reduces EXACTLY to a two-level
**hierarchical Kalman filter** — the standard practical implementation of
predictive coding:

```
Level 2 (slow belief):   mu2_t = mu2_{t-1} + w_t        (random-walk drift belief)
Level 1 (fast, observed): r_t   = mu2_t + v_t             (observation = belief + noise)
```

At each step: **predict** today's return from the current belief,
**compare** against what actually happened (the prediction error), then
**update** the belief by precision-weighting that error. This literally
IS free energy minimization, step by step — not a fit performed after the
fact.

**Core thesis — and why this engine is structured differently from the
others in this suite:** a return that is highly surprising relative to
the model's own beliefs (large precision-weighted prediction error) is
anomalous, and anomalous moves are read as **contrarian** — favoring
reversion back toward belief, not confirmation of the move. Most other
engines here have a secondary diagnostic that CONFIRMS the primary
signal's direction; here the surprise component explicitly argues
**against** the most recent move.

**Signal:**

```
score = 0.50*surprise_signal + 0.25*belief_signal + 0.25*model_confidence
```

- `surprise_signal` — -clip(zscore of today's prediction error vs. this
  window's own error history): the primary, contrarian driver
- `belief_signal` — zscore of the current drift belief relative to its
  own trajectory: secondary, non-contrarian context
- `model_confidence` — 1/(1+max(0, excess kurtosis of standardized
  errors)): are this ticker's surprises well-behaved (Gaussian) or
  fat-tailed? A validity check on the Gaussian assumption itself, not
  just a generic fit score.

**Distinct from ENTROPY-ENGINE / INFO-BOTTLENECK / THERMO-BOTTLENECK:**
those are static information-theoretic summaries of a distribution or
representation. This is the only one built around an explicit generative
model with beliefs that are actually *updated*, observation by
observation, via precision-weighted prediction errors.
        """)

    for universe_name, uni_data in universes1.items():
        top_etfs = uni_data.get("top_etfs", [])
        if not top_etfs:
            continue
        st.markdown(
            f'<div class="uni-title">{universe_name.replace("_"," ").title()}</div>',
            unsafe_allow_html=True)
        cols = st.columns(3)
        for idx, etf in enumerate(top_etfs):
            with cols[idx]:
                st.markdown(f"""
<div class="etf-card">
  <div class="etf-ticker">{etf['ticker']}</div>
  <div class="etf-score">AI score = {etf['ai_score']:.4f}</div>
  <div class="etf-score">best window = {etf.get('best_window','N/A')}d</div>
  <div class="etf-score">surprise = {etf.get('surprise_signal', float('nan')):.2f}</div>
  <div class="etf-score">confidence = {etf.get('model_confidence', float('nan')):.2f}</div>
</div>
""", unsafe_allow_html=True)

        with st.expander(f"Full ranking — {universe_name}"):
            full = uni_data.get("full_scores", {})
            if full:
                rows = []
                for t, info in full.items():
                    rows.append({
                        "ETF": t,
                        "AI Score": info.get("score"),
                        "Best Window (d)": info.get("best_window", "N/A"),
                        "Belief Forecast": info.get("belief_forecast"),
                        "Surprise Signal": info.get("surprise_signal"),
                        "Belief Signal": info.get("belief_signal"),
                        "Model Confidence": info.get("model_confidence"),
                        "Prediction Error Today": info.get("prediction_error_today"),
                    })
                df = pd.DataFrame(rows).sort_values("AI Score", ascending=False)
                st.dataframe(df, use_container_width=True, hide_index=True)
        st.divider()

    st.caption(
        f"Run date: {data1.get('run_date','?')} · "
        "Friston — Variational Free Energy / Predictive Coding · "
        "Scores are cross-sectional z-scores.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🔍 Explore Active Inference Rankings by Window")

    if not universes2:
        st.warning("Window-level detail not found. Re-run trainer.")
        st.stop()

    all_wins = set()
    for ud in universes2.values():
        all_wins.update(ud.get("windows", {}).keys())
    win_options = sorted([int(w) for w in all_wins])

    if not win_options:
        st.error("No window data available.")
        st.stop()

    default_idx  = win_options.index(252) if 252 in win_options else 0
    selected_win = st.selectbox(
        "Select lookback window",
        options=win_options,
        index=default_idx,
        format_func=lambda w: f"{w}d  (~{round(w/21)} months)",
    )
    win_key = str(selected_win)

    with st.expander("Window guidance", expanded=False):
        st.markdown("""
- **63d** — few observations for the belief to settle; reactive, noisier surprise history
- **126d** — 6-month window; recommended minimum for a stable error distribution
- **252d** — 1-year window; most stable belief trajectory; recommended primary signal
- **504d** — 2-year window; may blend multiple regimes into one slow-belief estimate
        """)

    st.markdown(f"### Active Inference Rankings at **{selected_win}d** window")

    for universe_name in ["FI_COMMODITIES", "EQUITY_SECTORS", "COMBINED"]:
        label = {
            "FI_COMMODITIES": "🏦 FI & Commodities",
            "EQUITY_SECTORS": "📈 Equity Sectors",
            "COMBINED":       "🌐 Combined",
        }.get(universe_name, universe_name)

        st.markdown(f'<div class="uni-title">{label}</div>', unsafe_allow_html=True)

        uni_data = universes2.get(universe_name, {})
        win_data = uni_data.get("windows", {}).get(win_key)

        if not win_data:
            st.info(f"No data for {universe_name} at {selected_win}d.")
            st.divider()
            continue

        cols = st.columns(3)
        for idx, etf in enumerate(win_data.get("top_etfs", [])):
            with cols[idx]:
                st.markdown(f"""
<div class="win-card">
  <div class="etf-ticker">{etf['ticker']}</div>
  <div class="etf-score">AI score = {etf['ai_score']:.4f}</div>
  <div class="etf-score">window = {selected_win}d</div>
  <div class="etf-score">surprise = {etf.get('surprise_signal', float('nan')):.2f}</div>
  <div class="etf-score">confidence = {etf.get('model_confidence', float('nan')):.2f}</div>
</div>
""", unsafe_allow_html=True)

        with st.expander(f"Full ranking — {label} @ {selected_win}d"):
            rows = win_data.get("full_ranking", [])
            if rows:
                df = pd.DataFrame(
                    rows,
                    columns=["ETF", "AI Score", "Belief Forecast", "Surprise Signal",
                             "Belief Signal", "Model Confidence"],
                )
                df.insert(0, "Rank", range(1, len(df) + 1))
                st.dataframe(df, use_container_width=True, hide_index=True)

        st.divider()

    st.caption(f"Window: {selected_win}d · Run date: {data2.get('run_date','?')}")
