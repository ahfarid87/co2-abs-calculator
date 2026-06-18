"""
CO₂ Adsorption Prediction & Optimisation Calculator
====================================================
Usage:  streamlit run app.py
Place all model files in the same folder as this script.
"""

import warnings, os
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from scipy.optimize import differential_evolution
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CO₂ Abs Calculator",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.0rem; font-weight: 700;
        background: linear-gradient(90deg, #1e3a5f, #2563EB);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .result-box {
        background: linear-gradient(135deg, #e8f4fd, #dbeafe);
        border-left: 5px solid #2563EB;
        padding: 1.2rem 1.5rem; border-radius: 0.5rem; margin: 0.5rem 0;
    }
    .result-big { font-size: 2.4rem; font-weight: 800; color: #1e3a5f; }
    .section-header {
        font-size: 1.1rem; font-weight: 600; color: #1e3a5f;
        border-bottom: 2px solid #bfdbfe; padding-bottom: 0.3rem; margin-bottom: 0.8rem;
    }
    .stAlert p { font-size: 0.9rem; }
    div[data-testid="metric-container"] {
        background: #f0f9ff; border-radius: 0.4rem;
        padding: 0.5rem; border: 1px solid #bae6fd;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Feature / bound definitions
# ─────────────────────────────────────────────────────────────────────────────
SYNTH_FEATURES   = ["PS", "AR", "DR", "C", "H", "N", "AT", "HR", "Ht"]
TEXTURE_FEATURES = ["BET", "TPV", "MPV"]
ALL_FEATURES     = SYNTH_FEATURES + ["T"]
INTEGER_VARS     = ["PS", "AR", "DR", "AT", "HR", "Ht", "T"]
SEED             = 42

FEATURE_META = {
    # name   : (label,                       min,    max,    default, step,  unit)
    "PS" : ("Particle Size (PS)",             50.0,   500.0,  127.5,  1.0,  "μm"),
    "AR" : ("Ratio of activating agent to carbon (AR)",               1.0,    10.0,    2.0,  1.0,  "—"),
    "DR" : ("Not Doped (0)or doped (1)",                0,    1,    1,  1.0,  "-"),
    "C"  : ("Carbon Content (C)",             50.0,   100.0,   80.0,  0.1,  "wt%"),
    "H"  : ("Hydrogen Content (H)",            0.0,    10.0,    1.0,  0.1,  "wt%"),
    "N"  : ("Nitrogen Content (N)",            0.0,    10.0,    1.0,  0.1,  "wt%"),
    "AT" : ("Activation Temperature (AT)",   500.0,  1000.0,  700.0,  1.0,  "°C"),
    "HR" : ("Heating Rate (HR)",               1.0,    20.0,    5.0,  1.0,  "°C/min"),
    "Ht" : ("Hold Time (Ht)",                  0.0,   120.0,   60.0,  1.0,  "min"),
    "T"  : ("Operating Temperature (T)",      0.0,   100.0,   25.0,  0.5,  "°C"),
}

# ─────────────────────────────────────────────────────────────────────────────
# Model loading  (cached so it runs once)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading saved models …")
def load_models():
    missing = []

    required = [
        "xgb_BET.json", "xgb_TPV.json", "xgb_MPV.json",
        "scaler_BET.pkl", "scaler_TPV.pkl", "scaler_MPV.pkl",
        "xgb_stage2.json", "scaler_stage2.pkl",
    ]
    for f in required:
        if not os.path.exists(f):
            missing.append(f)

    if missing:
        return None, missing

    try:
        xgb_BET = xgb.XGBRegressor(); xgb_BET.load_model("xgb_BET.json")
        xgb_TPV = xgb.XGBRegressor(); xgb_TPV.load_model("xgb_TPV.json")
        xgb_MPV = xgb.XGBRegressor(); xgb_MPV.load_model("xgb_MPV.json")

        models = {
            "prop_models":  {"BET": xgb_BET, "TPV": xgb_TPV, "MPV": xgb_MPV},
            "prop_scalers": {
                "BET": joblib.load("scaler_BET.pkl"),
                "TPV": joblib.load("scaler_TPV.pkl"),
                "MPV": joblib.load("scaler_MPV.pkl"),
            },
            "xgb_stage2":    xgb.XGBRegressor(),
            "scaler_stage2": joblib.load("scaler_stage2.pkl"),
        }
        models["xgb_stage2"].load_model("xgb_stage2.json")
        return models, []
    except Exception as e:
        return None, [str(e)]


# ─────────────────────────────────────────────────────────────────────────────
# Prediction pipeline
# ─────────────────────────────────────────────────────────────────────────────
def predict_abs(sample_dict, models):
    df_new = pd.DataFrame([sample_dict])

    for prop in TEXTURE_FEATURES:
        X_prop = models["prop_scalers"][prop].transform(df_new[SYNTH_FEATURES].values)
        df_new[f"{prop}_pred"] = models["prop_models"][prop].predict(X_prop)

    df_new["C_H_ratio"] = df_new["C"] / (df_new["H"] + 1e-6)
    df_new["AT_HR"]     = df_new["AT"] * df_new["HR"]
    df_new["BET_pred_T"]= df_new["BET_pred"] / (df_new["T"] + 1e-6)

    S2_FEATURES = (
        SYNTH_FEATURES + ["T"]
        + ["BET_pred", "TPV_pred", "MPV_pred"]
        + ["C_H_ratio", "AT_HR", "BET_pred_T"]
    )

    X  = df_new[S2_FEATURES].values
    Xs = models["scaler_stage2"].transform(X)
    abs_pred = float(models["xgb_stage2"].predict(Xs)[0])

    return {
        **sample_dict,
        "BET_pred": float(df_new["BET_pred"].iloc[0]),
        "TPV_pred": float(df_new["TPV_pred"].iloc[0]),
        "MPV_pred": float(df_new["MPV_pred"].iloc[0]),
        "Abs_pred": abs_pred,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Optimisation
# ─────────────────────────────────────────────────────────────────────────────
def run_optimisation(models, fixed_values, vary_features,
                     bounds_mode="percentile", maxiter=80, popsize=12):
    """
    Maximise predicted Abs using differential evolution.
    bounds_mode: 'percentile' → 5th-95th pct from reference  |  'manual' → use user bounds
    """
    bounds = []
    for feat in vary_features:
        lo, hi = bounds_mode[feat]          # caller passes resolved (lo,hi) pairs
        if np.isclose(lo, hi):
            lo, hi = lo - 1e-6, hi + 1e-6
        bounds.append((lo, hi))

    medians = {f: float(np.median([lo, hi])) for f, (lo, hi) in bounds_mode.items()}

    def objective(x_vec):
        candidate = dict(medians)
        candidate.update(fixed_values)
        for feat, val in zip(vary_features, x_vec):
            candidate[feat] = val
        for feat in INTEGER_VARS:
            if feat in candidate:
                candidate[feat] = int(round(candidate[feat]))
        pred = predict_abs(candidate, models)
        return -pred["Abs_pred"]

    result = differential_evolution(
        objective, bounds=bounds,
        seed=SEED, maxiter=maxiter, popsize=popsize,
        polish=True, updating="deferred", workers=1,
    )

    best = dict(medians)
    best.update(fixed_values)
    for feat, val in zip(vary_features, result.x):
        best[feat] = val
    for feat in INTEGER_VARS:
        if feat in best:
            best[feat] = int(round(best[feat]))

    best_pred = predict_abs(best, models)
    return best, best_pred, result


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — model loading status
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    #st.image("logo.png", width=120)
    st.image("logo.png", use_container_width=True)
    st.markdown("## CO₂ Abs Calculator")
    st.markdown("---")
    models, missing = load_models()
    if models:
        st.success("✅ All models loaded")
    else:
        st.error("⚠️ Model files missing")
        for m in missing:
            st.code(m)
        st.info(
            "Copy the following files into the **same folder** as `app.py`:\n\n"
            "`xgb_BET.json`  `xgb_TPV.json`  `xgb_MPV.json`\n\n"
            "`scaler_BET.pkl`  `scaler_TPV.pkl`  `scaler_MPV.pkl`\n\n"
            "`xgb_stage2.json`  `scaler_stage2.pkl`"
        )
    st.markdown("---")
    st.caption("CO2 Absorption Prediciton on Activated Carbon")


# ─────────────────────────────────────────────────────────────────────────────
# Main area
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🧪 CO₂ Adsorption Prediction & Optimisation</p>',
            unsafe_allow_html=True)

tab_pred, tab_opt = st.tabs(["🔮  Predict Abs", "⚙️  Optimise Synthesis"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ═══════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.markdown("Enter synthesis and operating conditions to predict CO₂ absorption.")
    st.markdown("---")

    # ── Input form ───────────────────────────────────────────────────────────
    col_synth, col_ops = st.columns([2, 1])

    inputs = {}

    with col_synth:
        st.markdown('<p class="section-header">🔧 Synthesis Parameters</p>',
                    unsafe_allow_html=True)
        g1, g2, g3 = st.columns(3)
        pairs = list(FEATURE_META.items())
        synth_pairs = [(k, v) for k, v in pairs if k in SYNTH_FEATURES]

        for i, (feat, (label, lo, hi, default, step, unit)) in enumerate(synth_pairs):
            col = [g1, g2, g3][i % 3]
            with col:
                # All args cast to float to avoid Streamlit mixed-type error
                val = st.number_input(
                    f"{feat}  ({unit})",
                    min_value=float(lo),
                    max_value=float(hi),
                    value=float(default),
                    step=float(step),
                    help=label,
                    key=f"pred_{feat}",
                )
                inputs[feat] = float(val)

    with col_ops:
        st.markdown('<p class="section-header">🌡️ Operating Condition</p>',
                    unsafe_allow_html=True)
        feat = "T"
        label, lo, hi, default, step, unit = FEATURE_META[feat]
        val = st.number_input(
            f"{feat}  ({unit})",
            min_value=float(lo),
            max_value=float(hi),
            value=float(default),
            step=float(step),
            help=label,
            key=f"pred_{feat}",
        )
        inputs[feat] = float(val)

        st.markdown("---")
        predict_btn = st.button("🔮 Predict Abs", use_container_width=True, type="primary")

    # ── Output ───────────────────────────────────────────────────────────────
    if predict_btn:
        if not models:
            st.error("Models not loaded. Check sidebar for instructions.")
        else:
            with st.spinner("Running two-stage prediction …"):
                result = predict_abs(inputs, models)

            st.markdown("---")
            st.markdown('<p class="section-header">📊 Prediction Results</p>',
                        unsafe_allow_html=True)

            # Main result
            st.markdown(
                f'<div class="result-box">'
                f'<div style="color:#555;font-size:0.9rem">Predicted CO₂ Absorption</div>'
                f'<div class="result-big">{result["Abs_pred"]:.4f}</div>'
                f'<div style="color:#555;font-size:0.85rem">mmol/g  </div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Intermediate textural predictions
            st.markdown("**Predicted Textural Properties (Stage 1 outputs):**")
            tc1, tc2, tc3 = st.columns(3)
            tc1.metric("BET Surface Area", f'{result["BET_pred"]:.2f}', "m²/g")
            tc2.metric("Total Pore Volume (TPV)", f'{result["TPV_pred"]:.4f}', "cm³/g")
            tc3.metric("Micro Pore Volume (MPV)", f'{result["MPV_pred"]:.4f}', "cm³/g")

            # Summary table
            with st.expander("📋 Full input + output summary"):
                summary = {k: [v] for k, v in inputs.items()}
                summary["BET_pred"]  = [round(result["BET_pred"], 3)]
                summary["TPV_pred"]  = [round(result["TPV_pred"], 4)]
                summary["MPV_pred"]  = [round(result["MPV_pred"], 4)]
                summary["Abs_pred"]  = [round(result["Abs_pred"], 4)]
                st.dataframe(pd.DataFrame(summary), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — OPTIMISATION
# ═══════════════════════════════════════════════════════════════════════════
with tab_opt:
    st.markdown(
        "Fix whichever parameters you want to hold constant, "
        "and the optimizer will find the synthesis conditions that **maximise** CO₂ absorption."
    )
    st.markdown("---")

    # ── Step 1: which variables to fix ──────────────────────────────────────
    st.markdown('<p class="section-header">Step 1 — Choose variables to FIX</p>',
                unsafe_allow_html=True)

    all_opt_vars = SYNTH_FEATURES + ["T"]
    fixed_choices = st.multiselect(
        "Select variables to hold fixed (leave empty to optimise everything):",
        options=all_opt_vars,
        default=[],
        key="opt_fixed_choices",
    )

    fixed_values = {}
    if fixed_choices:
        st.markdown("**Set fixed values:**")
        fc_cols = st.columns(min(len(fixed_choices), 4))
        for i, feat in enumerate(fixed_choices):
            label, lo, hi, default, step, unit = FEATURE_META[feat]
            with fc_cols[i % 4]:
                val = st.number_input(
                    f"{feat} ({unit})",
                    min_value=float(lo),
                    max_value=float(hi),
                    value=float(default),
                    step=float(step),
                    help=label,
                    key=f"fix_{feat}",
                )
                fixed_values[feat] = float(val)
                if feat in INTEGER_VARS:
                    fixed_values[feat] = int(round(fixed_values[feat]))

    vary_features = [f for f in all_opt_vars if f not in fixed_values]

    if vary_features:
        st.markdown(f"**Variables to optimise:** `{'`, `'.join(vary_features)}`")
    else:
        st.warning("All variables are fixed — nothing to optimise.")

    st.markdown("---")

    # ── Step 2: bounds for free variables ───────────────────────────────────
    st.markdown('<p class="section-header">Step 2 — Set search bounds</p>',
                unsafe_allow_html=True)

    bounds_mode = st.radio(
        "Bound source:",
        ["Use feature default ranges (recommended)", "Set custom bounds manually"],
        horizontal=True,
        key="opt_bounds_mode",
    )

    resolved_bounds = {}   # {feat: (lo, hi)}
    for feat in vary_features:
        label, lo, hi, default, step, unit = FEATURE_META[feat]
        resolved_bounds[feat] = (float(lo), float(hi))

    if bounds_mode == "Set custom bounds manually" and vary_features:
        with st.expander("Custom bounds for free variables"):
            for feat in vary_features:
                label, lo, hi, default, step, unit = FEATURE_META[feat]
                bc1, bc2 = st.columns(2)
                with bc1:
                    clo = st.number_input(
                        f"{feat} min ({unit})",
                        value=float(lo), step=float(step),
                        key=f"blo_{feat}",
                    )
                with bc2:
                    chi = st.number_input(
                        f"{feat} max ({unit})",
                        value=float(hi), step=float(step),
                        key=f"bhi_{feat}",
                    )
                resolved_bounds[feat] = (float(clo), float(chi))

    st.markdown("---")

    # ── Step 3: optimiser settings ──────────────────────────────────────────
    st.markdown('<p class="section-header">Step 3 — Optimiser settings</p>',
                unsafe_allow_html=True)

    oc1, oc2 = st.columns(2)
    with oc1:
        maxiter = st.slider("Max iterations", min_value=20, max_value=300, value=20, step=10)
    with oc2:
        popsize = st.slider("Population size", min_value=5, max_value=30, value=5, step=1)

    st.markdown("---")

    # ── Run optimisation ────────────────────────────────────────────────────
    run_opt = st.button(
        "⚙️ Run Optimisation",
        use_container_width=True,
        type="primary",
        disabled=(not vary_features),
    )

    if run_opt:
        if not models:
            st.error("Models not loaded. Check sidebar for instructions.")
        else:
            progress_bar = st.progress(0, text="Initialising optimiser …")

            with st.spinner(f"Optimising {len(vary_features)} variables — this may take ~30 s …"):
                progress_bar.progress(10, text="Running differential evolution …")
                best, best_pred, de_result = run_optimisation(
                    models=models,
                    fixed_values=fixed_values,
                    vary_features=vary_features,
                    bounds_mode=resolved_bounds,
                    maxiter=maxiter,
                    popsize=popsize,
                )
            progress_bar.progress(100, text="Done!")

            st.markdown("---")
            st.markdown('<p class="section-header">🏆 Optimisation Results</p>',
                        unsafe_allow_html=True)

            # Main result
            st.markdown(
                f'<div class="result-box">'
                f'<div style="color:#555;font-size:0.9rem">Best Predicted CO₂ Absorption</div>'
                f'<div class="result-big">{best_pred["Abs_pred"]:.4f}</div>'
                f'<div style="color:#555;font-size:0.85rem">Converged: '
                f'{"✅ Yes" if de_result.success else "⚠️ Not fully"} '
                f'| Iterations: {de_result.nit}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Textural properties
            st.markdown("**Predicted Textural Properties at Optimal Conditions:**")
            tc1, tc2, tc3 = st.columns(3)
            tc1.metric("BET", f'{best_pred["BET_pred"]:.2f}', "m²/g")
            tc2.metric("TPV", f'{best_pred["TPV_pred"]:.4f}', "cm³/g")
            tc3.metric("MPV", f'{best_pred["MPV_pred"]:.4f}', "cm³/g")

            st.markdown("**Optimal synthesis conditions:**")

            # Split fixed vs optimised for display
            opt_rows = []
            for feat in all_opt_vars:
                label, lo, hi, default, step, unit = FEATURE_META[feat]
                opt_rows.append({
                    "Feature": feat,
                    "Description": label,
                    "Unit": unit,
                    "Status": "Fixed" if feat in fixed_values else "Optimised",
                    "Value": best_pred.get(feat, best.get(feat, "—")),
                })

            opt_df = pd.DataFrame(opt_rows)
            # Colour-code status
            st.dataframe(
                opt_df.style.apply(
                    lambda col: ["background-color: #dbeafe" if v == "Optimised"
                                 else "background-color: #fef9c3" for v in col],
                    subset=["Status"],
                ),
                use_container_width=True,
                hide_index=True,
            )

            # Download
            csv_out = opt_df.copy()
            csv_out["Abs_pred"] = best_pred["Abs_pred"]
            csv_out["BET_pred"] = best_pred["BET_pred"]
            csv_out["TPV_pred"] = best_pred["TPV_pred"]
            csv_out["MPV_pred"] = best_pred["MPV_pred"]
            st.download_button(
                "⬇️ Download results as CSV",
                data=csv_out.to_csv(index=False),
                file_name="optimised_synthesis.csv",
                mime="text/csv",
            )
