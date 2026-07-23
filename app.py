# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model_loader import (
    load_and_prepare_data, train_models, create_explainer,
    load_artifacts, get_unscaled_data, _unwrap_pipeline
)

st.set_page_config(layout="wide", page_title="Heart Disease XAI – Hybrid")

st.title("🩺 Heart Disease Risk – Hybrid AI with XAI")
st.caption("Best-in-class student project: calibrated model, SHAP explanations, and what-if simulation.")

# ======== Train or Load ========
with st.spinner("Loading data and model..."):
    X_train, X_test, y_train, y_test, pipe_base, feature_names = load_and_prepare_data(".")
    try:
        model, features, threshold = load_artifacts()
    except Exception:
        model, threshold, leaderboard = train_models(X_train, y_train, feature_names)

# ======== Sidebar Inputs ========
st.sidebar.header("Patient Inputs")
X_unscaled = get_unscaled_data()

def slider_num(col, label):
    mn, mx = float(np.nanpercentile(X_unscaled[col], 1)), float(np.nanpercentile(X_unscaled[col], 99))
    default = float(np.nanmedian(X_unscaled[col]))
    step = 0.1 if col in ("oldpeak",) else 1.0
    return st.sidebar.slider(label, mn, mx, default, step=step)

user = {}
user["age"] = slider_num("age","Age")
user["sex"] = st.sidebar.selectbox("Sex (0=F,1=M)", [0,1], index=1)
user["cp"] = st.sidebar.selectbox("Chest Pain Type (0-3)", [0,1,2,3], index=0)
user["trestbps"] = slider_num("trestbps","Resting BP (mm Hg)")
user["chol"] = slider_num("chol","Serum Cholesterol (mg/dl)")
user["fbs"] = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1], index=0)
user["restecg"] = st.sidebar.selectbox("Resting ECG (0-2)", [0,1,2], index=0)
user["thalach"] = slider_num("thalach","Max Heart Rate")
user["exang"] = st.sidebar.selectbox("Exercise Induced Angina", [0,1], index=0)
user["oldpeak"] = slider_num("oldpeak","ST Depression")
user["slope"] = st.sidebar.selectbox("Slope of ST Segment (0-2)", [0,1,2], index=0)
user["ca"] = st.sidebar.selectbox("Major Vessels Colored (0-3)", [0,1,2,3], index=0)
user["thal"] = st.sidebar.selectbox("Thal (0=normal,1=fixed,2=reversible,3=unknown)", [0,1,2,3], index=2)

input_df = pd.DataFrame([user])[feature_names]

# ======== Predict ========
proba = model.predict_proba(input_df)[0,1]
try:
    _, _, threshold = load_artifacts()
except Exception:
    threshold = 0.5
pred = int(proba >= threshold)

left, right = st.columns(2)
with left:
    st.metric("Predicted Risk (probability)", f"{proba:.2f}")
with right:
    st.metric("Decision at threshold", "High Risk" if pred==1 else "Low Risk", delta=f"thr={threshold:.2f}")

st.divider()

# ======== SHAP Explanation ========
# ======== SHAP Explanation ========
# ======== SHAP Explanation ========
# ======== SHAP Explanation ========
st.subheader("🔍 Explanation (SHAP)")

import numpy as _np
import shap as _shap
import matplotlib.pyplot as _plt

def _scalar_base(x):
    arr = _np.array(x).reshape(-1)
    return float(arr[-1])  # use the only element or class-1 if available

def _one_row(sv):
    """Return a single (n_features,) vector from any SHAP return shape."""
    if isinstance(sv, list):
        sv = sv[-1] if len(sv) > 1 else sv[0]        # prefer positive class
    sv = _np.array(sv)
    if sv.ndim == 0:
        return sv.reshape(1)
    if sv.ndim == 1:
        return sv
    # shapes like (n_classes, n_samples, n_features) or (n_samples, n_features)
    if sv.ndim == 3:
        sv = sv[-1]          # take last class
    return sv[0]              # first sample → (n_features,)

def _align(values_vec, data_vec, names):
    """Ensure same length across values, data, names."""
    m = min(len(values_vec), len(data_vec), len(names))
    return values_vec[:m], data_vec[:m], names[:m]

def _plot_single_explanation(base_value_scalar, shap_values_row, feature_values_row, feature_names):
    shap_values_row = _np.ravel(shap_values_row)
    feature_values_row = _np.ravel(feature_values_row)
    shap_values_row, feature_values_row, feature_names = _align(shap_values_row, feature_values_row, feature_names)
    try:
        exp = _shap.Explanation(
            values=shap_values_row,
            base_values=base_value_scalar,
            data=feature_values_row,
            feature_names=feature_names
        )
        fig = _plt.figure()
        _shap.plots.waterfall(exp, max_display=len(feature_names), show=False)
        st.pyplot(fig)
    except Exception:
        # Fallback: bar plot per sample
        fig = _plt.figure()
        _shap.plots.bar(exp, max_display=len(feature_names), show=False)
        st.pyplot(fig)

try:
    base_pipe = _unwrap_pipeline(model)
    imputer = base_pipe.named_steps["imputer"]
    scaler  = base_pipe.named_steps["scaler"]
    base_clf = base_pipe.named_steps["model"]

    explainer, _ = create_explainer(model, X_train)

    if hasattr(base_clf, "feature_importances_"):
        # Tree model → explain transformed features
        x_trans = scaler.transform(imputer.transform(input_df))   # (1, n_features)
        sv = explainer.shap_values(x_trans)
        sv_row = _one_row(sv)
        base_val = _scalar_base(explainer.expected_value)
        _plot_single_explanation(base_val, sv_row, x_trans[0], feature_names)
    else:
        # Linear / KernelExplainer on raw features
        sv = explainer.shap_values(input_df, nsamples=200)
        sv_row = _one_row(sv)
        base_val = _scalar_base(explainer.expected_value)
        _plot_single_explanation(base_val, sv_row, input_df.iloc[0].values, feature_names)

except Exception as e:
    st.info(f"SHAP plot will be available after first successful training. ({e})")




# ======== What-if simulator ========
st.subheader("🧪 What-if Simulator")
cols = st.columns(4)
sim = user.copy()
sim["cp"] = cols[0].selectbox("cp", [0,1,2,3], index=sim["cp"])
sim["trestbps"] = cols[1].number_input("trestbps", value=float(sim["trestbps"]))
sim["chol"] = cols[2].number_input("chol", value=float(sim["chol"]))
sim["thalach"] = cols[3].number_input("thalach", value=float(sim["thalach"]))
sim_df = pd.DataFrame([sim])[feature_names]
sim_proba = model.predict_proba(sim_df)[0,1]
delta = sim_proba - proba
st.write(f"Original: **{proba:.2f}** → Simulated: **{sim_proba:.2f}**  (Δ {delta:+.2f})")
st.caption("Use this to see how lifestyle or clinical parameters might change risk (illustrative only).")
